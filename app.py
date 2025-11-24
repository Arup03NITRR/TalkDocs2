import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.prompts import PromptTemplate  # <--- NEW IMPORT
from htmlTemplates import css, bot_template, user_template
import os
import time
import sqlite3
import hashlib

# Pinecone Imports
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Constants
INDEX_NAME = "talkdocs"

# ==========================================
# 1. AUTHENTICATION & DATABASE LAYER
# ==========================================

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return True
    return False

def create_usertable():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT PRIMARY KEY, password TEXT)')
    conn.commit()
    conn.close()

def add_userdata(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    conn.close()
    return data

# ==========================================
# 2. DOCUMENT PROCESSING LAYER
# ==========================================

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        except Exception:
            st.error(f"Error reading {pdf.name}")
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# ==========================================
# 3. ISOLATED VECTOR STORE LAYER
# ==========================================

def get_vectorstore(user_namespace):
    """
    Connects to the Pinecone Index but restricts view to the user_namespace.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    pc = Pinecone(api_key=api_key)
    
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        return None

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=user_namespace 
    )
    return vectorstore

def process_and_upload(text_chunks, user_namespace):
    api_key = os.getenv("PINECONE_API_KEY")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    pc = Pinecone(api_key=api_key)
    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        with st.spinner(f"Initializing Knowledge Base Server..."):
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(10) 

    PineconeVectorStore.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace=user_namespace
    )

# ==========================================
# 4. CHAT CHAIN LAYER (STRICT PROMPTING)
# ==========================================

def get_conversation_chain(vectorstore):
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0) # Set temp to 0 for strictness
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 10, "fetch_k": 50}
    )

    # --- DEFINE STRICT PROMPT TEMPLATE ---
    custom_template = """You are a strict assistant. You must answer the question based ONLY on the provided context below.
    
    Rules:
    1. If the answer is found in the context, provide a clear and concise answer.
    2. If the answer is NOT in the context, you must strictly say: "I'm sorry, I cannot find this information in the uploaded documents."
    3. Do not use your own internal knowledge or make up facts.
    
    Context:
    {context}
    
    Question: {question}
    
    Helpful Answer:"""

    custom_prompt = PromptTemplate(
        template=custom_template,
        input_variables=["context", "question"]
    )
    # -------------------------------------

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt} # <--- INJECT PROMPT HERE
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.error("⚠️ Chatbot not initialized. Please refresh or upload a document.")
        return

    response = st.session_state.conversation.invoke({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# ==========================================
# 5. UI PAGES
# ==========================================

def auth_page():
    st.title("TalkDocs Secure Login")
    choice = st.selectbox("Login/Signup", ["Login", "Sign Up"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            create_usertable()
            hashed_pswd = make_hashes(password)
            if login_user(username, hashed_pswd):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Logged in!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Invalid credentials")

    elif choice == "Sign Up":
        new_user = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Create Account"):
            create_usertable()
            if add_userdata(new_user, make_hashes(new_password)):
                st.success("Account created! Please go to Login.")
            else:
                st.error("Username already taken.")

def chat_page(username):
    st.header(f"TalkDocs (User: {username})")
    
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['conversation'] = None
        st.rerun()

    if st.session_state.conversation is None:
        vectorstore = get_vectorstore(user_namespace=username)
        if vectorstore:
            st.session_state.conversation = get_conversation_chain(vectorstore)

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                process_and_upload(text_chunks, user_namespace=username)
                
                vectorstore = get_vectorstore(user_namespace=username)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Done!")

# ==========================================
# 6. MAIN APP FLOW
# ==========================================

def main():
    load_dotenv()
    st.set_page_config(page_title="TalkDocs", page_icon="📚")
    
    try:
        st.write(css, unsafe_allow_html=True)
    except:
        pass

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if st.session_state.logged_in:
        chat_page(st.session_state.username)
    else:
        auth_page()

if __name__ == '__main__':
    main()