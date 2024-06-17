from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

class LanguageAssistant:
    def __init__(self, model_name="meta/llama3-70b-instruct"):
        self.llama_model = ChatNVIDIA(model=model_name, stream=True)
        self.system_prompt = "You are an AI language model that provides detailed and accurate information on various topics."
        self.human_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{input}")
        ])
        self.chain = self.human_prompt_template | self.llama_model | StrOutputParser()

    def invoke(self, text_input):
        input_message = {"input": text_input}
        result = self.chain.invoke(input_message)
        return result
