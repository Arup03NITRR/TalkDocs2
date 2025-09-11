import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
import time

def get_pdf_text(pdf_docs):
    text = ""
    unreadable_files = []
    empty_text_files = []

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            file_text = ""
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    file_text += extracted_text

            if file_text.strip():
                text += file_text
            else:
                empty_text_files.append(pdf.name)

        except Exception as e:
            unreadable_files.append(pdf.name)

    if unreadable_files:
        st.warning(f"⚠️ The following PDFs could not be read due to corruption or format issues: {', '.join(unreadable_files)}")
        return None

    if empty_text_files:
        st.warning(f"⚠️ The following PDFs had no extractable text: {', '.join(empty_text_files)}")
        return None

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


def get_vectorstore(text_chunks):
    if not text_chunks:
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    if not vectorstore:
        st.error("❌ Cannot create conversation chain: vectorstore is missing or invalid.")
        return None

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant"
    )

    # Define system message
    system_prompt = """You are a helpful AI assistant for 'TalkDocs'.
    Always answer based only on the provided documents.
    If the answer is not found in the documents, say "I couldn't find that in your files."
    Provide clear, concise, and well-formatted responses.
    Also you are able to compare document's contents and judge which one is better or best.
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{question}")
    ])

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_userinput(user_question):
    # Append user's message to chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append(("user", user_question))

    # Display all messages so far (user + bot)
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)

    # Show typing placeholder
    typing_placeholder = st.empty()
    typing_placeholder.markdown('''
        <div class="chat-message bot">
            <div class="avatar">
                <img src="https://media.istockphoto.com/id/1221348467/vector/chat-bot-ai-and-customer-service-support-concept-vector-flat-person-illustration-smiling.jpg?s=612x612&w=0&k=20&c=emMSOYb4jWIVQQBVpYvP9LzGwPXXhcmbpZHlE6wgR78=" alt="Bot Avatar">
            </div>
            <div class="message">Typing<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
        </div>
    ''', unsafe_allow_html=True)

    # Simulate delay
    time.sleep(1.5)

    # Get response from Langchain conversation
    response = st.session_state.conversation({'question': user_question})
    bot_reply = response['answer']

    # Remove typing animation and show actual bot message
    typing_placeholder.empty()

    # Append bot reply to chat history
    st.session_state.chat_history.append(("bot", bot_reply))

    # Show bot reply
    st.write(bot_template.replace("{{MSG}}", bot_reply), unsafe_allow_html=True)


def main():
    load_dotenv()
    os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING_v2"] = os.getenv("LANGSMITH_TRACING_v2")
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
    os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

    st.set_page_config(page_title="TalkDocs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("TalkDocs :books:")
    st.subheader("Chat with multiple PDFs")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("❌ No PDF uploaded. Please upload at least one PDF before processing.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error("❌ PDF processing stopped. No usable text found.")
                    return

                text_chunks = get_text_chunks(raw_text)
                if not text_chunks:
                    st.error("❌ No text chunks could be generated. Please try a different document.")
                    return

                vectorstore = get_vectorstore(text_chunks)
                if not vectorstore:
                    st.error("❌ Vectorstore creation failed.")
                    return

                conversation = get_conversation_chain(vectorstore)
                if not conversation:
                    st.error("❌ Failed to initialize conversation chain.")
                    return

                st.session_state.conversation = conversation

            st.success("✅ PDF(s) processed successfully! You can now start chatting.")


if __name__ == '__main__':
    main()
