from interpreter import interpreter
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os

load_dotenv()

# Paste your OpenAI API key below.
interpreter.llm.api_key = os.environ.get("OPENAI_API_KEY")

# ask for > Can you solve this equation? 10x + 14 = 21 / 3
# according to doc, it must show a code implementation; however, as it uses ChatGPT , it shows a text description of how to solve the equation.
interpreter.chat()