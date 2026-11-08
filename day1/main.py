import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")
client=Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"
role="user"
prompt="Suggest a name for my clothing company"
# SYSTEM
message_system={
    "role": "system",
    "content": "You are a brand manager who suggests name for my company. name should be in one word. suggest only one name"
}
# message me role and content
message={
    "role": role,
    "content": prompt
}
