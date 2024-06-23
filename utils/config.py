import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    config = {
        #"OPENAI_API_TYPE": "azure",
        #"OPENAI_API_VERSION": "2023-12-01-preview",
        #"AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }
    return config

