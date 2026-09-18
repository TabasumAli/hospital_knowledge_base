"""
app.py
------
Streamlit chat app for the Ayub Medical Complex knowledge base.

- Loads the FAISS index built by ingest.py
- Retrieves the most relevant chunks for each question
- Sends ONLY those chunks (never outside knowledge) to Groq's
  openai/gpt-oss-120b model, with a strict prompt that forbids
  answering from anything other than the provided context
- Displays the exact source PDF + page number for every answer,
  and clearly says "I don't know" instead of guessing when the
  knowledge base doesn't contain the answer
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4  # how many chunks to retrieve per question

NO_ANSWER_PHRASE = "I don't have that information in the knowledge base."

SYSTEM_PROMPT = f"""You are a factual assistant for Ayub Medical Complex.
Answer the user's question using ONLY the context below. Do not use any
outside knowledge and do not guess or infer anything not explicitly
stated in the context.

Rules:
1. If the answer is fully contained in the context, answer it clearly
   and precisely, staying faithful to the wording and facts given.
2. If the context does not contain the answer, respond with EXACTLY:
   "{NO_ANSWER_PHRASE}"
3. Never make up document names, page numbers, phone numbers,
   procedures, or policies.
4. Keep the answer concise and directly address the question.

Context:
{{context}}
"""


# ---------------------------------------------------------------------------
# Cached resources (loaded once per session, not per question)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_retriever():
    if not os.path.isdir(FAISS_INDEX_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # safe: this is our own local index
    )
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})


@st.cache_resource
def load_chain():
    retriever = load_retriever()
    if retriever is None:
        return None

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found. Add it to a .env file (see SETUP.md).")
        st.stop()

    llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{input}")]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain


def format_sources(source_documents):
    """Build a de-duplicated, human-readable source list from chunk metadata."""
    seen = set()
    lines = []
    for doc in source_documents:
        category = doc.metadata.get("category", "unknown")
        source_file = doc.metadata.get("source_file", "unknown.pdf")
        page = doc.metadata.get("page_number", "?")
        key = (category, source_file, page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- **{source_file}** ({category}), page {page}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Ayub Medical Complex Assistant", page_icon="🏥")
st.title("🏥 Ayub Medical Complex — Knowledge Base Assistant")
st.caption(
    "Answers are generated only from the hospital's own PDF documents. "
    "If something isn't in the knowledge base, the assistant will say so "
    "instead of guessing."
)

chain = load_chain()

if chain is None:
    st.warning(
        "No FAISS index found. Run `python ingest.py` first "
        "(see SETUP.md steps 1–4), then restart this app."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about admissions, emergency, departments, safety...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            result = chain.invoke({"input": question})
            answer = result["answer"]
            source_docs = result.get("context", [])

        st.markdown(answer)

        # Only show sources if the model actually answered (not the "I don't know" case)
        if source_docs and NO_ANSWER_PHRASE not in answer:
            with st.expander("📄 Sources"):
                st.markdown(format_sources(source_docs))

    st.session_state.messages.append({"role": "assistant", "content": answer})