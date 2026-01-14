from typing import List, Dict, Optional
import os
import time
import uuid
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from firecrawl import FirecrawlApp
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from fastembed import TextEmbedding
from agents import Agent, Runner
from openai import AsyncOpenAI

import nest_asyncio
import asyncio

# ---------------------------
# IMPORTANT CONFIG
# ---------------------------
load_dotenv()
nest_asyncio.apply()

COLLECTION_NAME = "docs_embeddings"


# ---------------------------
# Helpers
# ---------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split long documentation into smaller chunks for better embeddings/search."""
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def init_session_state():
    defaults = {
        "setup_complete": False,
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
        "firecrawl_api_key": os.getenv("FIRECRAWL_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "doc_url": "",
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "selected_voice": "coral",
        "pages_count": 0,
        "chunks_count": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------
# Qdrant Setup
# ---------------------------
def setup_qdrant_collection(qdrant_url: str, qdrant_api_key: str, collection_name: str = COLLECTION_NAME):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    embedding_model = TextEmbedding()
    test_embedding = list(embedding_model.embed(["test"]))[0]
    embedding_dim = len(test_embedding)

    # Create collection only if not exists
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

    return client, embedding_model


# ---------------------------
# Firecrawl Docs Crawl
# ---------------------------
def crawl_documentation(firecrawl_api_key: str, url: str, limit_pages: int = 5) -> List[Dict]:
    firecrawl = FirecrawlApp(api_key=firecrawl_api_key)

    pages = []

    response = firecrawl.crawl_url(
        url,
        params={
            "limit": limit_pages,
            "scrapeOptions": {"formats": ["markdown"]},
        },
    )

    while True:
        for page in response.get("data", []):
            content = page.get("markdown") or ""
            metadata = page.get("metadata", {}) or {}
            source_url = metadata.get("sourceURL", url)

            if not content.strip():
                continue

            pages.append(
                {
                    "content": content,
                    "url": source_url,
                    "metadata": {
                        "title": metadata.get("title", ""),
                        "description": metadata.get("description", ""),
                        "language": metadata.get("language", "en"),
                        "crawl_date": datetime.now().isoformat(),
                    },
                }
            )

        next_url = response.get("next")
        if not next_url:
            break

        response = firecrawl.get(next_url)
        time.sleep(1)

    return pages


# ---------------------------
# Store Embeddings (Chunks)
# ---------------------------
def store_embeddings(
    client: QdrantClient,
    embedding_model: TextEmbedding,
    pages: List[Dict],
    collection_name: str = COLLECTION_NAME,
):
    points = []
    total_chunks = 0

    for page in pages:
        content = page.get("content", "")
        if not content:
            continue

        chunks = chunk_text(content, chunk_size=1000, overlap=150)
        total_chunks += len(chunks)

        for i, chunk in enumerate(chunks):
            embedding = list(embedding_model.embed([chunk]))[0]

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk,
                        "chunk_index": i,
                        "url": page.get("url", ""),
                        **page.get("metadata", {}),
                    },
                )
            )

    # Upsert in batches (faster)
    batch_size = 50
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i : i + batch_size])

    return total_chunks


# ---------------------------
# Agent Setup
# ---------------------------
def setup_agents(openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key

    processor_agent = Agent(
        name="Documentation Processor",
        instructions="""
You are a helpful documentation assistant.
Your task:
1) Use ONLY the provided documentation context.
2) Answer clearly and concisely.
3) Use bullet points where helpful.
4) If something is missing, say you couldn't find it in docs.
5) At the end, show sources as clickable links.
Make the answer easy to speak out loud.
""",
        model="gpt-4o",
    )

    return processor_agent


# ---------------------------
# Query Processing
# ---------------------------
async def process_query(
    query: str,
    client: QdrantClient,
    embedding_model: TextEmbedding,
    processor_agent: Agent,
    collection_name: str,
    openai_api_key: str,
    voice: str,
):
    try:
        # Embed query
        query_embedding = list(embedding_model.embed([query]))[0]

        # Stable Qdrant search
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding.tolist(),
            limit=4,
            with_payload=True,
        )

        if not search_results:
            return {"status": "error", "error": "No relevant documents found in the vector database."}

        # Build context for GPT (limit text to avoid huge prompt)
        context = "You have access to the following documentation snippets:\n\n"
        sources = []

        for r in search_results:
            payload = r.payload or {}
            url = payload.get("url", "Unknown URL")
            content = payload.get("content", "")

            if url not in sources:
                sources.append(url)

            context += f"SOURCE: {url}\n"
            context += f"SNIPPET:\n{content[:1200]}\n\n"

        context += f"\nUSER QUESTION: {query}\n"
        context += "Answer now."

        # GPT Answer
        processor_result = await Runner.run(processor_agent, context)
        answer_text = processor_result.final_output

        # TTS
        async_openai = AsyncOpenAI(api_key=openai_api_key)
        audio_response = await async_openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=answer_text,
            response_format="mp3",
        )

        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"voice_response_{uuid.uuid4()}.mp3")

        with open(audio_path, "wb") as f:
            f.write(audio_response.content)

        return {
            "status": "success",
            "text_response": answer_text,
            "audio_path": audio_path,
            "sources": sources,
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------
# Sidebar UI
# ---------------------------
def sidebar_config():
    with st.sidebar:
        st.title("🔑 Configuration")
        st.markdown("---")

        st.session_state.qdrant_url = st.text_input(
            "Qdrant URL",
            value=st.session_state.qdrant_url,
            placeholder="https://YOUR-CLUSTER.qdrant.io",
        )
        st.session_state.qdrant_api_key = st.text_input(
            "Qdrant API Key",
            value=st.session_state.qdrant_api_key,
            type="password",
        )
        st.session_state.firecrawl_api_key = st.text_input(
            "Firecrawl API Key",
            value=st.session_state.firecrawl_api_key,
            type="password",
        )
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
        )

        st.markdown("---")
        st.session_state.doc_url = st.text_input(
            "Documentation URL",
            value=st.session_state.doc_url,
            placeholder="https://docs.example.com",
        )

        st.markdown("---")
        st.markdown("### 🎤 Voice Settings")

        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        st.session_state.selected_voice = st.selectbox(
            "Select Voice",
            options=voices,
            index=voices.index(st.session_state.selected_voice),
        )

        st.markdown("---")
        limit_pages = st.slider("Crawl Pages Limit", min_value=1, max_value=20, value=5)

        if st.button("🚀 Initialize System", type="primary"):
            required = [
                st.session_state.qdrant_url,
                st.session_state.qdrant_api_key,
                st.session_state.firecrawl_api_key,
                st.session_state.openai_api_key,
                st.session_state.doc_url,
            ]

            if not all(required):
                st.error("❌ Please fill all required fields.")
                return

            try:
                with st.spinner("🔄 Connecting to Qdrant + loading embeddings model..."):
                    client, embedding_model = setup_qdrant_collection(
                        st.session_state.qdrant_url,
                        st.session_state.qdrant_api_key,
                        COLLECTION_NAME,
                    )
                    st.session_state.client = client
                    st.session_state.embedding_model = embedding_model

                with st.spinner("🔄 Crawling documentation pages..."):
                    pages = crawl_documentation(
                        st.session_state.firecrawl_api_key,
                        st.session_state.doc_url,
                        limit_pages=limit_pages,
                    )
                    st.session_state.pages_count = len(pages)

                with st.spinner("🔄 Creating chunks + storing embeddings in Qdrant..."):
                    chunks_count = store_embeddings(
                        st.session_state.client,
                        st.session_state.embedding_model,
                        pages,
                        COLLECTION_NAME,
                    )
                    st.session_state.chunks_count = chunks_count

                with st.spinner("🔄 Setting up AI agent..."):
                    processor_agent = setup_agents(st.session_state.openai_api_key)
                    st.session_state.processor_agent = processor_agent

                st.session_state.setup_complete = True
                st.success("✅ System initialized successfully!")

            except Exception as e:
                st.session_state.setup_complete = False
                st.error(f"❌ Setup failed: {str(e)}")


# ---------------------------
# Main App
# ---------------------------
def run_streamlit():
    st.set_page_config(
        page_title="Customer Support Voice Agent",
        page_icon="🎙️",
        layout="wide",
    )

    init_session_state()
    sidebar_config()

    st.title("🎙️ Customer Support Voice Agent")
    st.caption("Ask questions from documentation and get both text + voice replies.")

    if st.session_state.setup_complete:
        st.success(
            f"✅ Ready! Pages: {st.session_state.pages_count} | Chunks Stored: {st.session_state.chunks_count}"
        )
    else:
        st.info("👈 Configure and Initialize from the sidebar to start.")

    st.markdown("---")

    query = st.text_input(
        "💬 Ask a question about the documentation:",
        placeholder="e.g., How do I authenticate API requests?",
        disabled=not st.session_state.setup_complete,
    )

    if query and st.session_state.setup_complete:
        with st.status("Processing your query...", expanded=True) as status:
            try:
                status.update(label="🔎 Searching docs + generating response...", state="running")

                result = asyncio.get_event_loop().run_until_complete(
                    process_query(
                        query=query,
                        client=st.session_state.client,
                        embedding_model=st.session_state.embedding_model,
                        processor_agent=st.session_state.processor_agent,
                        collection_name=COLLECTION_NAME,
                        openai_api_key=st.session_state.openai_api_key,
                        voice=st.session_state.selected_voice,
                    )
                )

                if result["status"] == "success":
                    status.update(label="✅ Done!", state="complete")

                    st.subheader("📌 Answer")
                    st.write(result["text_response"])

                    st.subheader(f"🔊 Voice Response ({st.session_state.selected_voice})")
                    st.audio(result["audio_path"], format="audio/mp3")

                    with open(result["audio_path"], "rb") as f:
                        st.download_button(
                            label="📥 Download MP3",
                            data=f.read(),
                            file_name=f"voice_response_{st.session_state.selected_voice}.mp3",
                            mime="audio/mp3",
                        )

                    st.subheader("🔗 Sources")
                    for s in result["sources"]:
                        st.markdown(f"- {s}")

                else:
                    status.update(label="❌ Error", state="error")
                    st.error(result.get("error", "Unknown error"))

            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    run_streamlit()
