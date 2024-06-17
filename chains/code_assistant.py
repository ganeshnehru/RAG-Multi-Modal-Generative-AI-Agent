from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

class CodeAssistant:
    def __init__(self, model_name="ibm/granite-34b-code-instruct", temperature = 0):
        self.chat_model = ChatNVIDIA(model=model_name)
        self.system_prompt = "You are an AI model specialized in providing coding assistance, including code generation, explanation, and debugging."
        self.human_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{input}")
        ])
        self.chain = self.human_prompt_template | self.chat_model | StrOutputParser()

    def invoke(self, text_input):
        input_message = {"input": text_input}
        result = self.chain.invoke(input_message)
        return result
