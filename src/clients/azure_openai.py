import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

def get_azure_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    if not endpoint or not api_key or not api_version:
        raise ValueError("Azure OpenAI env vars missing. Check AZURE_OPENAI_ENDPOINT/KEY/API_VERSION")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
