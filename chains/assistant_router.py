# import re
# import logging
# from chains.code_assistant import CodeAssistant
# from chains.language_assistant import LanguageAssistant
# from chains.vision_assistant import VisionAssistant
#
# class AssistantRouter:
#     def __init__(self):
#         self.code_assistant = CodeAssistant()
#         self.language_assistant = LanguageAssistant()
#         self.vision_assistant = VisionAssistant()
#
#     def route_input(self, user_input='', image_path=None):
#         """
#         Route the input to the appropriate assistant based on the content of the user input.
#
#         :param user_input: str, The input text from the user.
#         :param image_path: str, Path to an image file if provided.
#         :return: tuple, The response from the appropriate assistant and the assistant name.
#         """
#         try:
#             if image_path:
#                 # Process image and route to VisionAssistant
#                 image_b64 = self.vision_assistant.process_image(image_path)
#                 if image_b64 is None:
#                     raise ValueError("Failed to process image.")
#                 input_string = f"{user_input}|{image_b64}"
#                 response = self.vision_assistant.invoke(input_string)
#                 return response, 'VisionAssistant'
#
#             if self.is_code_related(user_input):
#                 response = self.code_assistant.invoke(user_input)
#                 return response, 'CodeAssistant'
#             else:
#                 response = self.language_assistant.invoke(user_input)
#                 return response, 'LanguageAssistant'
#         except Exception as e:
#             logging.error(f"Error in AssistantRouter.route_input: {e}")
#             return {"error": str(e)}, 'Error'
#
#     def is_code_related(self, text):
#         """
#         Determine if the text input is related to coding.
#
#         :param text: str, The input text.
#         :return: bool, True if the text is code related, False otherwise.
#         """
#         # Basic keyword-based detection
#         code_keywords = [
#             'function', 'class', 'def', 'import', 'print', 'variable',
#             'loop', 'array', 'list', 'dictionary', 'exception', 'error', 'bug',
#             'code', 'compile', 'execute', 'algorithm', 'data structure' , 'java' , 'python' , 'javascript', 'c++',
#             'c#', 'ruby', 'php', 'html', 'css', 'sql', 'swift', 'kotlin', 'go', 'rust', 'typescript', 'r', 'perl',
#             'scala', 'shell', 'bash', 'powershell', 'objective-c', 'matlab', 'groovy', 'lua', 'dart', 'cobol',
#             'fortran', 'haskell', 'lisp', 'pascal', 'prolog', 'scheme', 'smalltalk', 'verilog', 'vhdl',
#             'assembly', 'coffeescript', 'f#', 'julia', 'racket', 'scratch', 'solidity', 'vba', 'abap', 'apex',
#             'awk', 'clojure', 'd', 'elixir', 'erlang', 'forth', 'hack', 'idris', 'j', 'julia', 'kdb+', 'labview',
#             'logtalk', 'lolcode', 'mumps', 'nim', 'ocaml', 'pl/i', 'postscript', 'powershell', 'rpg', 'sas', 'sml',
#             'tcl', 'turing', 'unicon', 'x10', 'xquery', 'zsh'
#         ]
#         pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in code_keywords) + r')\b', re.IGNORECASE)
#         return bool(pattern.search(text))



import re
import logging
from chains.code_assistant import CodeAssistant
from chains.language_assistant import LanguageAssistant
from chains.vision_assistant import VisionAssistant

class AssistantRouter:
    def __init__(self):
        self.code_assistant = CodeAssistant()
        self.language_assistant = LanguageAssistant()
        self.vision_assistant = VisionAssistant()

    def route_input(self, user_input='', image_path=None):
        """
        Route the input to the appropriate assistant based on the content of the user input.

        :param user_input: str, The input text from the user.
        :param image_path: str, Path to an image file if provided.
        :return: tuple, The response from the appropriate assistant and the assistant name.
        """
        try:
            if image_path:
                # Process image and route to VisionAssistant
                image_b64 = self.vision_assistant.process_image(image_path)
                if image_b64 is None:
                    raise ValueError("Failed to process image.")
                input_string = f"{user_input}|{image_b64}"
                response = self.vision_assistant.invoke(input_string)
                return response, 'VisionAssistant'

            if self.is_code_related(user_input):
                response = self.code_assistant.invoke(user_input)
                return response, 'CodeAssistant'
            else:
                response = self.language_assistant.invoke(user_input)
                return response, 'LanguageAssistant'
        except Exception as e:
            logging.error(f"Error in AssistantRouter.route_input: {e}")
            return {"content": f"Error: {str(e)}"}, 'Error'

    def is_code_related(self, text):
        """
        Determine if the text input is related to coding.

        :param text: str, The input text.
        :return: bool, True if the text is code related, False otherwise.
        """
        # Basic keyword-based detection
        code_keywords = [
            'function', 'class', 'def', 'import', 'print', 'variable',
            'loop', 'array', 'list', 'dictionary', 'exception', 'error', 'bug',
            'code', 'compile', 'execute', 'algorithm', 'data structure', 'java', 'python', 'javascript', 'c++',
            'c#', 'ruby', 'php', 'html', 'css', 'sql', 'swift', 'kotlin', 'go', 'rust', 'typescript', 'r', 'perl',
            'scala', 'shell', 'bash', 'powershell', 'objective-c', 'matlab', 'groovy', 'lua', 'dart', 'cobol',
            'fortran', 'haskell', 'lisp', 'pascal', 'prolog', 'scheme', 'smalltalk', 'verilog', 'vhdl',
            'assembly', 'coffeescript', 'f#', 'julia', 'racket', 'scratch', 'solidity', 'vba', 'abap', 'apex',
            'awk', 'clojure', 'd', 'elixir', 'erlang', 'forth', 'hack', 'idris', 'j', 'julia', 'kdb+', 'labview',
            'logtalk', 'lolcode', 'mumps', 'nim', 'ocaml', 'pl/i', 'postscript', 'powershell', 'rpg', 'sas', 'sml',
            'tcl', 'turing', 'unicon', 'x10', 'xquery', 'zsh'
        ]
        pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in code_keywords) + r')\b', re.IGNORECASE)
        return bool(pattern.search(text))
