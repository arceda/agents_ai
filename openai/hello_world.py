from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os

# en windows es necesario especificar la ruta del .env
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
        "role": "user", 
        "content": "Act as an AI assitant and Explain machine learning to a 5 years old in 3 differents ways",
        "max_tokens": 20,
        "temperature": 0.8
    }
  ]
)
message = completion.choices[0].message.content

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]
while True:
    message = input("User: ")
    if message == "end": break
    
    if message:
        messages.append({ "role":"user", "content":message })
        chat = client.chat.completions.create(model="gpt-3.5-turbo",  messages = messages)

    reply = chat.choices[0].message.content
    print("ChatGPT reply:", reply)
    messages.append({ "role":"user", "content":reply })
