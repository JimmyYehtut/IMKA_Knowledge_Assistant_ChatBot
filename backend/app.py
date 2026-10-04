import hashlib
import os
import tempfile

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import streamlit as st
from dotenv import load_dotenv

from ingest.embed_and_store import ingest_files
from ingest.load_documents import SUPPORTED_EXTENSIONS
from rag.retrieval_chain import build_retrieval_chain

load_dotenv()

DEPARTMENTS = ["PLG", "HMG", "CRG", "LEGAL", "IAG", "SHARED"]

# --- Page Config & Styling ---
st.set_page_config(
    page_title="RAG Data Pipeline Dashboard",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563EB !important;
        color: white !important;
        border-radius: 0.375rem !important;
        font-weight: 500 !important;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Header ---
st.markdown("<div class='main-header'>🧠 RAG Data Pipeline & Chat Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Upload files (PDF, DOCX, TXT), tag them with department access, index into Qdrant, and chat instantly.</div>",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=90)
    st.header("Pipeline Configuration")

    openai_key = os.getenv("OPENAI_API_KEY", "")

    try:
        from rag.qdrant_init import get_qdrant_client
        get_qdrant_client().get_collections()
        qdrant_reachable = True
    except Exception:
        qdrant_reachable = False

    st.markdown("### Service Connections")
    st.markdown("🟢 **OpenAI**" if openai_key else "🔴 **OpenAI**: Missing Key")
    st.markdown("🟢 **Qdrant**" if qdrant_reachable else "🔴 **Qdrant**: Unreachable (is `docker compose up` running?)")

    st.divider()
    st.markdown("### Settings")
    index_name = st.text_input("Qdrant Collection Name", value=os.getenv("QDRANT_COLLECTION_NAME", "ragindex"))
    embedding_model = st.text_input("Embedding Model", value=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    chat_model = st.text_input("Chat Model", value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1"))

    st.divider()
    st.info("💡 Settings load from `.env` automatically. Sidebar changes override for the current session.")

# --- Main Layout ---
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.subheader("📁 Data Ingestion Pipeline")
    st.markdown("Upload files, set department access, then click **Start Ingest Pipeline**.")

    uploaded_files = st.file_uploader(
        "Choose documents (PDF, Office, HTML, CSV, text, ...)",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown(f"**Selected files ({len(uploaded_files)}):**")
        for f in uploaded_files:
            st.text(f"• {f.name} ({round(f.size / 1024, 2)} KB)")

    # --- Department Metadata ---
    st.markdown("#### 🏢 Department Access Metadata")
    owner_department = st.selectbox(
        "Owner Department",
        options=DEPARTMENTS,
        help="The department that owns this document.",
    )
    allowed_departments = st.multiselect(
        "Allowed Departments",
        options=DEPARTMENTS,
        default=[owner_department],
        help="Departments that can retrieve this document. Must include the owner.",
    )
    if owner_department not in allowed_departments:
        allowed_departments = [owner_department] + allowed_departments

    classification = st.selectbox(
        "Classification",
        options=["Internal", "Confidential", "Restricted", "Public"],
        help="Document sensitivity level.",
    )
    document_type = st.selectbox(
        "Document Type",
        options=["Policy", "Report", "Guideline", "Procedure", "General"],
    )
    version = st.text_input("Version", value="1.0")

    if uploaded_files and st.button("Start Ingest Pipeline"):
        if not openai_key or not qdrant_reachable:
            st.error("❌ OPENAI_API_KEY must be set and Qdrant must be reachable before ingestion.")
        elif not allowed_departments:
            st.error("❌ Select at least one allowed department.")
        else:
            with st.spinner("Executing data pipeline..."):
                try:
                    temp_paths = []
                    original_names = []
                    temp_dir = tempfile.gettempdir()

                    for uploaded_file in uploaded_files:
                        file_hash = hashlib.md5(uploaded_file.name.encode("utf-8")).hexdigest()
                        ext = os.path.splitext(uploaded_file.name)[1]
                        temp_path = os.path.join(temp_dir, f"uploaded_{file_hash}{ext}")
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        temp_paths.append(temp_path)
                        original_names.append(uploaded_file.name)

                    os.environ["QDRANT_COLLECTION_NAME"] = index_name
                    os.environ["OPENAI_EMBEDDING_MODEL"] = embedding_model
                    os.environ["OPENAI_CHAT_MODEL"] = chat_model

                    result = ingest_files(
                        temp_paths,
                        metadata={
                            "owner_department": owner_department,
                            "allowed_departments": ",".join(allowed_departments),
                            "classification": classification,
                            "document_type": document_type,
                        },
                        version=version,
                        original_names=original_names,
                    )
                    chunks_created = result["total_chunks"] if result else 0

                    if chunks_created:
                        st.success(
                            f"🎉 Ingested **{chunks_created}** chunks into Qdrant collection **{index_name}**.\n\n"
                            f"**Owner:** {owner_department} | "
                            f"**Allowed:** {', '.join(allowed_departments)} | "
                            f"**Class:** {classification}"
                        )
                    else:
                        st.warning("⚠️ No chunks created. Check your document formatting.")

                    for path in temp_paths:
                        if os.path.exists(path):
                            os.remove(path)

                except Exception as e:
                    st.error(f"❌ Pipeline failed: {e}")

    with st.expander("🛠️ View Pipeline Details"):
        st.markdown(
            """
            **How the data pipeline works:**
            1. **Load**: PyPDF / Docx2txt / TextLoader parse the uploaded file.
            2. **Chunk**: `RecursiveCharacterTextSplitter` — `chunk_size=1000`, `overlap=200`.
            3. **Metadata**: Each chunk is tagged with `document_id`, `document_name`,
               `owner_department`, `allowed_departments`, `classification`, `document_type`, `version`.
            4. **Embed**: OpenAI `text-embedding-3-small` (1024 dimensions).
            5. **Store**: Upserted into the Qdrant collection.

            The `allowed_departments` field enables RBAC filtering at retrieval time:
            only users whose department appears in the list can retrieve the chunk.
            """
        )

with col2:
    st.subheader("💬 RAG Interactive Chat")

    if st.session_state.messages:
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    user_query = st.chat_input("Ask a question about your uploaded documents...")

    if user_query:
        with chat_container:
            with st.chat_message("user"):
                st.write(user_query)

        st.session_state.messages.append({"role": "user", "content": user_query})

        with chat_container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                try:
                    os.environ["QDRANT_COLLECTION_NAME"] = index_name
                    os.environ["OPENAI_EMBEDDING_MODEL"] = embedding_model
                    os.environ["OPENAI_CHAT_MODEL"] = chat_model

                    chain = build_retrieval_chain()
                    for chunk in chain.stream(user_query):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")

                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    placeholder.markdown("❌ Error. Check API keys and index setup.")
