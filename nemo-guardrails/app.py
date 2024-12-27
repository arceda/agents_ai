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


@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400   

    try:    
        guardrails = getGuardrails(data)
    except Exception as e:        
        print(f"An error occurred initializing Guardrails: {e}")
        response = jsonify({'error': 'Failed to initialize Guardrails config. Plase Check config (yaml & colang)'})
        response.status_code = 500
        return response

        
    messages, last_message_content = process_messages(data.get("messages", []))

    prompt = ChatPromptTemplate.from_messages(messages)

    
    tools = data.get("tools")

    llm = ChatOpenAI(
        temperature=data.get("temperature", 0),
        max_tokens=data.get("max_tokens", 1024),
        streaming=data.get("stream", False),
        model_name=data.get("model", "gpt-3.5-turbo"),
    )

    if tools:
        llm.model_kwargs = {"tools": tools}

    output_parser = StrOutputParser()

    use_guardrails = data.get('guardrails')

    if use_guardrails:
        print("## Using guardrails")
        chain = prompt | llm | output_parser
        chain_with_guardrails = guardrails | chain
    else:
        chain = prompt | llm
        chain_with_guardrails = chain

    response_aimessage = chain_with_guardrails.invoke({"input": last_message_content})

    tools_calls = None
    response_content = ""

    if isinstance(response_aimessage, AIMessage):    
        tools_calls = response_aimessage.additional_kwargs.get("tool_calls")
        response_content = response_aimessage.content
    else:
        response_content = response_aimessage
        if "output" in response_content:
            response_content = response_content["output"]

    if tools_calls:
        return jsonify(
            {
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": data.get("model", "gpt-3.5-turbo"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "function_call": tools_calls[0].get("function"),
                        },
                        "logprobs": None,
                        "finish_reason": "function_call",
                    }
                ],
            }
        )
    else:
        return jsonify(
            {
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": data.get("model", "gpt-3.5-turbo"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content,
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
            }
        )    


def getGuardrails(data):
    guardrails_config_input = data.get("guardrails")

    # Initialize colang_content and yaml_content
    colang_content = None
    yaml_content = None

    if guardrails_config_input:
        colang_content = guardrails_config_input.get("colang")
        yaml_content = guardrails_config_input.get("yaml")

    if colang_content or yaml_content:
         # Create a new event loop
        loop = asyncio.new_event_loop()
        # Set the created event loop as the current event loop
        asyncio.set_event_loop(loop)
        print("Guardrails: Using user config")
        local_config = RailsConfig.from_content(
            colang_content=colang_content, yaml_content=yaml_content
        )
        guardrails = RunnableRails(config=local_config)
    else:
        guardrails = default_guardrails

    return guardrails


def process_messages(data_messages):
    messages = []
    last_user_message_index = None
    last_user_content = None

    for index, msg in enumerate(data_messages):
        content = msg.get("content", "")
        role = msg.get("role", None)

        if role == "user":
            last_user_content = content
            messages.append(HumanMessage(content=content))
            last_user_message_index = index
        elif role in ["bot", "assistant"]:
            tool_calls = msg.get("tool_calls")
            if tool_calls is not None:
                messages.append(
                    AIMessage(
                        content=content, additional_kwargs={"tool_calls": tool_calls}
                    )
                )
            else:
                messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "tool":
            tool_call_id = msg.get("tool_call_id")
            messages.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        else:
            raise ValueError(f"Unknown message type {role}")

    if last_user_message_index is not None:
        messages[last_user_message_index] = HumanMessagePromptTemplate.from_template(
            "{input}"
        )

    #print('messages:', messages)
    return messages, last_user_content


@app.route("/status", methods=["GET"])
def status():
    return {"status": "running"}


if __name__ == "__main__":
    app.run()
