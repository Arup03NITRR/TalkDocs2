import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template
import os
import time

# Pinecone Imports
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Constants
INDEX_NAME = "talkdocs"

def get_pdf_text(pdf_docs):
    text = ""
    unreadable_files = []
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        except Exception as e:
            unreadable_files.append(pdf.name)

    if unreadable_files:
        st.warning(f"⚠️ Could not read: {', '.join(unreadable_files)}")
    
    return text


def get_text_chunks(text):
    if not text:
        return []

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore_connection():
    """
    Attempts to connect to an existing Pinecone index.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        pc = Pinecone(api_key=api_key)
        existing_indexes = [index.name for index in pc.list_indexes()]

        if INDEX_NAME in existing_indexes:
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embeddings
            )
            return vectorstore
        else:
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None


def add_to_vectorstore(text_chunks):
    """
    Adds new text chunks to the Pinecone index.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    try:
        pc = Pinecone(api_key=api_key)
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if INDEX_NAME not in existing_indexes:
            with st.spinner(f"Creating new index '{INDEX_NAME}'..."):
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(2)

        vectorstore = PineconeVectorStore.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            index_name=INDEX_NAME
        )
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ Error updating Pinecone: {str(e)}")
        return None


def get_conversation_chain(vectorstore):
    if not vectorstore:
        return None

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer' 
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 20, "fetch_k": 50} 
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True 
    )

    return conversation_chain


def handle_userinput(user_question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append(("user", user_question))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)

    typing_placeholder = st.empty()
    typing_placeholder.markdown('<div class="chat-message bot">Thinking...</div>', unsafe_allow_html=True)

    try:
        if st.session_state.conversation:
            # --- FIX APPLIED HERE ---
            # Replaced direct call with .invoke() to fix DeprecationWarning
            response = st.session_state.conversation.invoke({'question': user_question})
            bot_reply = response['answer']
        else:
            bot_reply = "⚠️ I am not connected to the Knowledge Base. Please check your API keys or process a document."

        typing_placeholder.empty()
        st.session_state.chat_history.append(("bot", bot_reply))
        st.write(bot_template.replace("{{MSG}}", bot_reply), unsafe_allow_html=True)
    
    except Exception as e:
        typing_placeholder.empty()
        st.error(f"Error: {str(e)}")


def main():
    load_dotenv()
    
    st.set_page_config(page_title="TalkDocs", page_icon=":books:")
    
    try:
        st.write(css, unsafe_allow_html=True)
    except NameError:
        pass

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Auto-Connect Logic
    if st.session_state.conversation is None:
        existing_store = get_vectorstore_connection()
        if existing_store:
            st.session_state.conversation = get_conversation_chain(existing_store)

    st.header("TalkDocs :books:")
    st.subheader("Ask questions from your Database")

    user_question = st.text_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Add new documents")
        pdf_docs = st.file_uploader(
            "Upload PDFs to add to the database", accept_multiple_files=True
        )

        if st.button("Process & Add"):
            if not pdf_docs:
                st.error("Please upload a PDF.")
                return

            with st.spinner("Processing & Uploading to Pinecone..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.stop()

                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    st.stop()

                vectorstore = add_to_vectorstore(text_chunks)
                
                if vectorstore:
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("✅ Added to Knowledge Base!")

if __name__ == '__main__':
    main()