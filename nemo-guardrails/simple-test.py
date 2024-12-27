import time
import uuid
import json
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Importing necessary modules for the chat functionality
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import asyncio

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Load and set up the guardrails configuration
config_path = "./config"
default_rails_config = RailsConfig.from_path(config_path)
default_guardrails = RunnableRails(config=default_rails_config)


llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

chain.invoke({"input": "What is the main advantage of writing documentation in a Jupyter notebook? Respond with one sentence."})