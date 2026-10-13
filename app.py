import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import Settings

load_dotenv(dotenv_path="/Users/revathy_santhosh/rag-project/.env")

st.set_page_config(page_title="RAG Q&A", page_icon="📄")
st.title("📄 Document Q&A (RAG)")
st.caption("Ask questions about: HBR Apple article, NID B.Des admissions, NYIT international student guide, Edward Guiliano Fellowship report, AI Engineering RAG chapter")

@st.cache_resource(show_spinner="Loading documents and building index (first run only)...")
def load_query_engine():
    import os
    from dotenv import load_dotenv
    from llama_index.core import Settings
    from llama_index.llms.groq import Groq
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    load_dotenv(dotenv_path="/Users/revathy_santhosh/rag-project/.env")

    Settings.llm = Groq(model="openai/gpt-oss-20b", api_key=os.getenv("GROQ_API_KEY"))
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    print("LLM and embedding model configured.")

    import glob
    import fitz  # this is PyMuPDF's import name
    from llama_index.core import Document

    documents = []
    for pdf_path in glob.glob("/Users/revathy_santhosh/rag-project/data/*.pdf"):
        pdf = fitz.open(pdf_path)
        file_name = pdf_path.split("/")[-1]
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():  # skip blank pages
                documents.append(Document(text=text, metadata={"file_name": file_name, "page": page_num}))

    print(f"Loaded {len(documents)} page-documents from {len(set(d.metadata['file_name'] for d in documents))} files")
    for fname in sorted(set(d.metadata["file_name"] for d in documents)):
        total_chars = sum(len(d.text) for d in documents if d.metadata["file_name"] == fname)
        print(f"- {fname}: {total_chars} chars")

    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    chroma_client = chromadb.PersistentClient(path="/Users/revathy_santhosh/rag-project/chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("rag_project")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )   
    print("Index built and stored in Chroma.")

    query_engine = index.as_query_engine(similarity_top_k=3)

    response = query_engine.query("What does the HBR article say about how Apple organizes for innovation?")
    print(response)

    print("\n--- Sources used ---")
    for node in response.source_nodes:
        print(f"- {node.metadata.get('file_name')} (page {node.metadata.get('page')}) — score: {node.score:.3f}")

    # ↓↓↓ PASTE YOUR EXACT CODE FROM rag_pipeline.ipynb HERE, IN THIS ORDER: ↓↓↓
    # 1. The Settings.llm = Groq(...) / Settings.embed_model = HuggingFaceEmbedding(...) cell
    # 2. The PyMuPDF document-loading loop (builds `documents`)
    # 3. The Chroma index-build cell (builds `index`)
    # 4. The line: query_engine = index.as_query_engine(similarity_top_k=3)
    # ↑↑↑ (same code you already copied into evaluate_rag.py — copy it again here) ↑↑↑

    return query_engine

query_engine = load_query_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about the documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(question)
            answer = str(response)
            st.write(answer)

            with st.expander("Sources used"):
                for i, node in enumerate(response.source_nodes, 1):
                    st.markdown(f"**Source {i}** (score: {node.score:.3f})")
                    st.text(node.text[:400] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})