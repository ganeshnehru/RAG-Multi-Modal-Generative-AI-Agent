from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from PIL import Image, ImageOps
import base64
import io

from dotenv import load_dotenv
load_dotenv()

class VisionAssistant:
    def __init__(self, model_name="microsoft/phi-3-vision-128k-instruct"):
        self.chat_model = ChatNVIDIA(model=model_name)
        self.system_prompt = """You are an AI vision assistant specialized in analyzing, 
                                describing and answering questions about images. You are accurately 
                                able to describe the contents of an image, including objects, actions, 
                                and scenes."""
        self.human_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "{input}")
        ])
        self.chain = self.human_prompt_template | self.chat_model | StrOutputParser()

    def process_image(self, image_path, desired_size=256):
        image = Image.open(image_path)
        image.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)
        image = ImageOps.pad(image, (desired_size, desired_size), color="white")
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def invoke(self, text_input, image_b64):
        input_message = [
            {"type": "text", "text": text_input},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]
        result = self.chat_model.invoke([HumanMessage(content=input_message)])
        return result.content
