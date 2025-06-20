from pydantic import BaseModel

class Prompt(BaseModel):
    """
    A model for the input prompt.
    
    Attributes:
        text (str): The input text provided by the user.
    """
    text: str