import os
import streamlit as st
import pickle
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from chains.language_assistant import LanguageAssistant
from chains.code_assistant import CodeAssistant
from chains.vision_assistant import VisionAssistant
from chains.models.whisper_asr import WhisperASR

# Load environment variables
load_dotenv()

# Set environment variable to avoid conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set Streamlit page config
st.set_page_config(page_title="Agent-Nesh 🤖", layout="wide")

# Page header
# st.markdown("<h1 style='text-align: left;'>Convo w/ Agent-Nesh! 🤖</h1>", unsafe_allow_html=True)

# Sidebar for settings and file upload
with st.sidebar:
    import streamlit as st

    st.markdown("<h1 style='text-align: center; font-size: 50px;'>Agent-Nesh 🤖</h1>", unsafe_allow_html=True)
    st.subheader("")

    st.subheader("About:")
    st.write("""
    Agent-Nesh is a RAG-based multi-modal generative AI chatbot that utilizes:
    - [Meta Llama 3](https://build.nvidia.com/explore/discover#llama3-70b)
    - [Microsoft Phi 3 Vision](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct)
    - [IBM Granite](https://build.nvidia.com/explore/discover#granite-34b-code-instruct)
    - [OpenAI Whisper](https://openai.com/research/whisper/)

    from [Nvidia NIM](https://www.nvidia.com/en-us/ai/), to provide intelligent, context-aware responses to text, image, code, and voice inputs. 
    By leveraging these multimodal architectures, Agent-Nesh can understand and respond to a wide array of user queries.
    """)
    st.subheader("")
    st.subheader("")

    st.subheader("Upload Files")
    uploaded_image = None

    uploaded_files = st.file_uploader("Input documents/images for analysis:", type=["png", "jpg", "jpeg", "pdf", "txt"],
                                      accept_multiple_files=True)

    DOCS_DIR = os.path.abspath("./uploaded_docs")
    os.makedirs(DOCS_DIR, exist_ok=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type.startswith("image/"):
                uploaded_image = uploaded_file
            else:
                file_path = os.path.join(DOCS_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                st.success(f"File {uploaded_file.name} uploaded successfully!")

    st.subheader("")
    st.subheader("")

    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Function to load and process documents
def load_and_process_documents(docs_dir, vector_store_path):
    raw_documents = DirectoryLoader(docs_dir).load()
    if raw_documents:
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)
        vectorstore = FAISS.from_documents(documents, document_embedder)
        with open(vector_store_path, "wb") as f:
            pickle.dump(vectorstore, f)
        return vectorstore
    else:
        return None

# Initialize embedding models
document_embedder = NVIDIAEmbeddings(model="NV-Embed-QA", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="NV-Embed-QA", model_type="query")

# Vector database store
vector_store_path = "../vectorstore.pkl"
vectorstore = None

if use_existing_vector_store == "Yes" and os.path.exists(vector_store_path):
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    st.sidebar.success("Existing vector store loaded successfully.")
else:
    with st.spinner("Processing documents..."):
        vectorstore = load_and_process_documents(DOCS_DIR, vector_store_path)
    st.sidebar.success("Vector store created and saved.")

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)

# Initialize the assistants
language_assistant = LanguageAssistant(model_name="meta/llama3-70b-instruct")
code_assistant = CodeAssistant(model_name="ibm/granite-34b-code-instruct")
vision_assistant = VisionAssistant(model_name="microsoft/phi-3-vision-128k-instruct")

# Initialize WhisperASR
whisper_asr = WhisperASR(model_name="base")

# Function to determine which assistant to use
def route_input(user_input, uploaded_image=None):
    code_keywords = ["code", "programming", "debug", "function", "class", "script", "algorithm"]
    if uploaded_image:
        return vision_assistant, "Phi"
    elif any(keyword in user_input.lower() for keyword in code_keywords):
        return code_assistant, "Granite"
    return language_assistant, "Llama"

# Main Content
with st.container():
    chat_container = st.container()

    with chat_container:
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    input_container = st.container()

    with input_container:
        st.subheader("")
        col1, col2 = st.columns([1, 8], gap="small")

        with col1:
            # Add a button for voice input with only a speech bubble icon emoji
            if st.button("💬 Voice Input"):
                with st.spinner("Listening..."):
                    user_input = whisper_asr.run()
                    st.success("Listening complete!")

        with col2:
            # Add a text input for manual user input
            text_input = st.chat_input("Enter your query here...")

        if text_input:
            user_input = text_input

if 'user_input' not in locals():
    user_input = ""

if user_input or uploaded_image:
    # Display user message in the chat
    if user_input:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

    assistant, model_name = route_input(user_input, uploaded_image)

    if assistant == vision_assistant and uploaded_image:
        # Display the uploaded image with a fixed width
        with chat_container:
            with st.chat_message("user"):
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=False, width=700)

        image_b64 = assistant.process_image(uploaded_image)
        augmented_user_input = f"Image and question: {user_input}" if user_input else "Analyze this image"
        input_message = [
            {"type": "text", "text": augmented_user_input},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]

        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for response in assistant.chat_model.stream([HumanMessage(content=input_message)]):
                    full_response += response.content
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(user_input)
        context = "\n\n".join(doc.page_content for doc in docs)
        augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"

        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                for response in assistant.chain.stream({"input": augmented_user_input}):
                    full_response += response
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
