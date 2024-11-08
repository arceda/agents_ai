from pydantic import BaseModel
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
import os
import json

# en windows es necesario especificar la ruta del .env
load_dotenv()

from pydantic import BaseModel
from openai import OpenAI

from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed # return a class instance

jsonstr1 = json.dumps(event.__dict__) 
print(jsonstr1)
