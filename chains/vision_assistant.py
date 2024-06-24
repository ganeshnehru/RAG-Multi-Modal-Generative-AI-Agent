from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from chains.memory import central_memory
from PIL import Image, ImageOps
import base64
import io
from dotenv import load_dotenv
import logging

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
        self.memory = central_memory

    # def process_image(self, image_path, desired_size=256):
    #     try:
    #         image = Image.open(image_path)
    #         image.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)
    #         image = ImageOps.pad(image, (desired_size, desired_size), color="white")
    #         buffered = io.BytesIO()
    #         image.save(buffered, format="PNG")
    #         return base64.b64encode(buffered.getvalue()).decode('utf-8')
    #     except Exception as e:
    #         logging.error(f"Error in VisionAssistant.process_image: {e}")
    #         return None

    def process_image(self, image_path, desired_size=256):
        try:
            # Open the input image
            with Image.open(image_path) as img:
                # Convert the image to PNG format
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                buffered.seek(0)
                image = Image.open(buffered)

                # Resize the image
                image.thumbnail((desired_size, desired_size), Image.Resampling.LANCZOS)
                image = ImageOps.pad(image, (desired_size, desired_size), color="white")

                # Save the processed image to a buffer
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")

                # Encode the image in base64
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error in VisionAssistant.process_image: {e}")
            return None

    def invoke(self, input_string):
        try:
            if '|' not in input_string:
                raise ValueError("Input must be in the format 'text|base64_image'.")
            text_input, image_b64 = input_string.split('|', 1)
            input_message = [
                {"type": "text", "text": text_input},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
            result = self.chat_model.invoke([HumanMessage(content=input_message)])
            self.add_to_memory(text_input, result.content)  # Save the interaction to memory
            return result.content
        except Exception as e:
            logging.error(f"Error in VisionAssistant.invoke: {e}")
            return {"error": str(e)}

    def add_to_memory(self, text_input, response):
        """
        Add the interaction to the memory.

        :param text_input: str, The input text from the user.
        :param response: str, The response from the assistant.
        """
        self.memory.save_context({'input': text_input}, {'response': response})

