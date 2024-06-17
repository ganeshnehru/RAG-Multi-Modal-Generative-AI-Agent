# Generative-AI-Digital-Assistant-w-RAG (Agent-Nesh 🤖)

Agent-Nesh is a Retrieval-Augmented Generation (RAG)-based multi-modal AI assistant that leverages advanced AI models to provide intelligent, context-aware responses to various types of input including text, images, code, and voice. This project uses the following models:

- [Meta Llama 3](https://build.nvidia.com/explore/discover#llama3-70b)
- [Microsoft Phi 3 Vision](https://build.nvidia.com/microsoft/phi-3-vision-128k-instruct)
- [IBM Granite](https://build.nvidia.com/explore/discover#granite-34b-code-instruct)
- [OpenAI Whisper](https://openai.com/research/whisper/)

## Features

- **Text Assistance**: Handle general text-based queries.
- **Code Assistance**: Provide coding assistance and help with code-related queries.
- **Image Analysis**: Analyze and describe images.
- **Voice Recognition**: Convert spoken language into text.

## Project Structure

```
Generative agent/
├── models/
│   ├── llama.py
│   ├── phi_vision.py
│   ├── granite.py
│   └── whisper_asr.py
├── chains/
│   ├── language_assistant.py
│   ├── code_assistant.py
│   └── vision_assistant.py
├── utils/
│   └── image_processor.py
├── agent/
│   ├── tools/
│   │   └── uml_to_code.py
│   ├── prompt_templates.py
│   └── llm_agent.py
└── app.py
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/agent-nesh.git
    cd agent-nesh
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your NVIDIA_API_KEY and OPENAI_API_KEY.

### Running the Application

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your browser and navigate to the provided URL to interact with Agent-Nesh.

## Usage

- **Text Queries**: Type your text queries in the provided input box and get responses from the language model.
- **Code Assistance**: Enter your coding queries to receive code assistance.
- **Image Analysis**: Upload images for analysis and description.
- **Voice Input**: Use the voice input feature to transcribe spoken language into text.


## Acknowledgements

- [NVIDIA NIM](https://www.nvidia.com/en-us/ai/)
- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)

