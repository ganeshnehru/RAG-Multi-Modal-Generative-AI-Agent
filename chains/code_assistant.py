from langchain.chains import ConversationChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from chains.memory import central_memory
from dotenv import load_dotenv
import logging

load_dotenv()

class CodeAssistant:
    def __init__(self, model_name="ibm/granite-34b-code-instruct"):
        self.model = ChatNVIDIA(model_name=model_name, stream=True)
        self.memory = central_memory
        self.chain = ConversationChain(
            llm=self.model,
            memory=self.memory,
            verbose=True
        )

    def invoke(self, text_input):
        try:
            if not isinstance(text_input, str) or not text_input.strip():
                raise ValueError("Input must be a non-empty string.")
            return self.chain.predict(input=text_input)
        except Exception as e:
            logging.error(f"Error in CodeAssistant.invoke: {e}")
            return {"error": str(e)}
