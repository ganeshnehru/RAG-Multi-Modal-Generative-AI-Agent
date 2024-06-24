
import streamlit as st
from PIL import Image
from chains.assistant_router import AssistantRouter
import time
from chains.models.whisper_asr import WhisperASR
from datetime import datetime
import uuid

# Initialize the AssistantRouter
router = AssistantRouter()

st.set_page_config(page_title="Agent-Nesh 🤖", layout="wide")

# Initialize the WhisperASR
whisper_asr = WhisperASR(model_name="base")

st.title("Ask me anything!")

# Initialize chat history
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for file upload and ASR
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 50px;'>Agent-Nesh 🤖</h1>", unsafe_allow_html=True)
    # st.subheader("About:")
    # st.write("""
    # Agent-Nesh is a RAG-based multi-modal generative AI chatbot that utilizes:
    # - [Meta Llama 3](https://build.nvidia.com/explore/discover#llama3-70b)
    # - [Microsoft Phi 3 Vision](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct)
    # - [IBM Granite](https://build.nvidia.com/explore/discover#granite-34b-code-instruct)
    # - [OpenAI Whisper](https://openai.com/research/whisper/)
    #
    # Developed using [Nvidia NIM](https://www.nvidia.com/en-us/ai/), Agent-Nesh provides intelligent, context-aware responses to text, image, code, and voice inputs.
    # """)

    st.markdown("---")
    st.subheader("Upload Files")

    # File upload section
    uploaded_file = st.file_uploader("Upload File", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_path = f"uploaded_image_{st.session_state.session_id}.png"
        image.save(image_path)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.markdown("---")

    # ASR button section
    if st.button("Record and Transcribe Audio"):
        with st.spinner("Recording..."):
            try:
                transcription = whisper_asr.run()
                st.session_state.transcription = transcription
            except Exception as e:
                st.error(f"Error during transcription: {e}")

# Display chat messages from history on app rerun
for message in sorted(st.session_state.messages, key=lambda x: x['timestamp']):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and isinstance(message["content"], dict):
            if "code" in message["content"]:
                st.code(message["content"]["code"], language=message["content"]["language"])
            else:
                st.markdown(message["content"]["text"])
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            st.markdown(f"*Model Used: {message['assistant_name']}*")

# Handle ASR transcription after processing
if "transcription" in st.session_state:
    transcription = st.session_state.pop("transcription")

    # Display user transcription in chat message container
    with st.chat_message("user"):
        st.markdown(transcription)

    # Check if there is an uploaded image
    try:
        if uploaded_file:
            combined_input = transcription
            response, assistant_name = router.route_input(combined_input, image_path)
        else:
            response, assistant_name = router.route_input(transcription)
    except Exception as e:
        st.error(f"Error during processing: {e}")
        response = {"content": "Sorry, an error occurred while processing your request.", "assistant_name": "Error"}

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        response_content = response if isinstance(response, str) else response.get("content", "")
        if "```" in response_content:
            parts = response_content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    full_response += part + " "
                    response_placeholder.markdown(full_response)
                else:
                    language, code = part.strip().split("\n", 1)
                    st.code(code, language=language.strip())
                time.sleep(0.05)
        else:
            for word in response_content.split():
                full_response += word + " "
                response_placeholder.markdown(full_response)
                time.sleep(0.05)
        st.markdown(f"*Model Used: {assistant_name}*")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "user", "content": transcription, "timestamp": datetime.now().isoformat()})
    st.session_state.messages.append(
        {"role": "assistant", "content": response_content, "assistant_name": assistant_name, "timestamp": datetime.now().isoformat()})

# Accept user input
text_input = st.chat_input("Enter your message:")

# Handle input and file upload processing
if uploaded_file and text_input:
    combined_input = text_input

    # Process input with image
    try:
        response, assistant_name = router.route_input(combined_input, image_path)
    except Exception as e:
        st.error(f"Error during processing: {e}")
        response = {"content": "Sorry, an error occurred while processing your request.", "assistant_name": "Error"}

    # Display user image and assistant response
    with st.chat_message("user"):
        st.markdown(combined_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        response_content = response if isinstance(response, str) else response.get("content", "")
        if "```" in response_content:
            parts = response_content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    full_response += part + " "
                    response_placeholder.markdown(full_response)
                else:
                    language, code = part.strip().split("\n", 1)
                    st.code(code, language=language.strip())
                time.sleep(0.05)
        else:
            for word in response_content.split():
                full_response += word + " "
                response_placeholder.markdown(full_response)
                time.sleep(0.05)
        st.markdown(f"*Model Used: {assistant_name}*")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "user", "content": combined_input, "timestamp": datetime.now().isoformat()})
    st.session_state.messages.append(
        {"role": "assistant", "content": response_content, "assistant_name": assistant_name, "timestamp": datetime.now().isoformat()})

    # Manually update the conversation memory for VisionAssistant
    if assistant_name == "VisionAssistant":
        router.vision_assistant.add_to_memory(combined_input, response_content)

# Handle text input without image
elif text_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": text_input, "timestamp": datetime.now().isoformat()})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(text_input)

    # Process input without image
    try:
        response, assistant_name = router.route_input(text_input)
    except Exception as e:
        st.error(f"Error during processing: {e}")
        response = {"content": "Sorry, an error occurred while processing your request.", "assistant_name": "Error"}

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        response_content = response if isinstance(response, str) else response.get("content", "")

        # Check if the response contains code blocks
        if "```" in response_content:
            parts = response_content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    full_response += part + " "
                    response_placeholder.markdown(full_response)
                else:
                    language, code = part.strip().split("\n", 1)
                    st.code(code, language=language.strip())
                time.sleep(0.05)
        else:
            for word in response_content.split():
                full_response += word + " "
                response_placeholder.markdown(full_response)
                time.sleep(0.05)
        st.markdown(f"*Model Used: {assistant_name}*")

    # Add assistant response to chat history
    st.session_state.messages.append(
        {"role": "assistant", "content": response_content, "assistant_name": assistant_name, "timestamp": datetime.now().isoformat()})
