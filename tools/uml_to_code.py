from models.granite import Granite
from models.temp.phi_vision_old import PhiVision

class UMLToCode:
    def __init__(self):
        self.phi_vision = PhiVision()
        self.granite = Granite()


    def get_uml_description(self, img_path, programming_language):

        # DEFINE PROMPT MESSAGES FOR PHI_VISION AND GRANITE MODELS.
        phi_message = """
            As a software engineer, your task is to analyze the provided UML diagram and write a comprehensive explanation of the classes and their relationships. Your explanation should cover the following aspects for each class:
    
            **Class Name**:
            - Provide the name of the class.
    
            **Attributes**:
            - List all attributes of the class, specifying whether each attribute is public or private.
    
            **Methods**:
            - Enumerate all methods of the class, indicating whether each method is public or private.
    
            **Relationships**:
            - Describe the relationships between classes, including associations, dependencies, inheritances, and aggregations.
    
            Your explanation should be detailed and structured, ensuring clarity and coherence. Use the provided template below to organize your analysis:
    
            **Class Name**:
            - [Class name goes here]
    
            **Attributes**:
            - [Attribute 1]: [Public/Private]
            - [Attribute 2]: [Public/Private]
            - ...
    
            **Methods**:
            - [Method 1]: [Public/Private]
            - [Method 2]: [Public/Private]
            - ...
    
            **Relationships**:
            - [Relationship 1]
            - [Relationship 2]
            - ...
    
            Your analysis will aid in understanding the structure and design of the software system represented by the UML diagram. Ensure clarity and completeness in your explanation.
            """

        # GETTING UML DESCRIPTION FROM THE IMAGE INPUT (PHI_VISION)
        uml_description = self.phi_vision.get_response(img_path, phi_message)

        granite_message = f'''You are an experienced software engineer. Given the following description of a UML diagram, you are required to write {programming_language} code that corresponds to the description: {uml_description}.
        You will first start off by explaining the description of the UML diagram, then proceed to write the corresponding code.'''


        # GETTING CODE FROM UML DESCRIPTION (GRANITE)
        code = self.granite.get_response(granite_message)

        return code.content


uc = UMLToCode()
img_path = '/Users/ganesh/Downloads/imgs/class-diagram-example.png'
programming_language = "Python"

result = uc.get_uml_description(img_path, programming_language)

print(result)