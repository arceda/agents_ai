# Saia with NeMO Guardrails

NeMo Guardrails is an innovative open-source toolkit designed to empower developers to implement programmable guardrails in LLM-based conversational applications seamlessly. These guardrails, also known as "rails," provide a robust mechanism for controlling the output of a large language model. They enable a range of functionalities, including content restrictions (e.g., avoiding political topics), custom response behaviors for specific user requests, adherence to predefined dialog paths, utilization of specific language styles, extraction of structured data, and much more.

## Overview

Saia, an OpenAI API Chat Wrapper, leverages the familiar OpenAI interface to enhance your conversational applications. It introduces a HTTP server that hosts the chat completion endpoint (`/v1/chat/completions`), simplifying the integration of advanced chat functionalities into your projects.


## Running Saia with NeMO Guardrails Locally for Development

### Step 1: Set Up Environment Variables
Create a `.env` file similar to the Docker setup to store your configuration settings.

### Step 2: Install Dependencies
Install the necessary Python dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Start the Application
Launch the application with the following command:

```bash
python app.py
```

### Step 4: Test Your Setup
Your local development server is ready. Test the chat completion endpoint via HTTP POST:

```
http://localhost:5000/v1/chat/completions
```

## Requirements
- Python 3.11 (3.12 not working)


## Running Saia with NeMO Guardrails Locally Using Docker

### Step 1: Set Up Environment Variables
First, create a `.env` file to store your configuration settings:

```plaintext
# Saia API Key
OPENAI_API_KEY= 

# Saia Proxy Base URL
OPENAI_BASE_URL=https://api.beta.saia.ai/proxy/openai/v1
```

### Step 2: Launch Using Docker Compose
Execute the following command to start your services using Docker:

```bash
docker compose up
```

### Step 3: Test Your Setup
Congratulations! Your server is now up and running. Test the chat completion endpoint via HTTP POST:

```
http://localhost:5000/v1/chat/completions
```

### Calling the API

Http POST to: http://localhost:5000/v1/chat/completions
```
{
    "model": "gpt-4",
    "messages": [    
        {
            "role": "system",
            "content": "You are a professional translator. Translate the user's text into English."
        },
        {
            "role": "user",
            "content": "Buenos días, hoy es un día soleado"
        }
    ],
    "stream": true,
    "temperature": 1,
    "max_tokens": 1000,
    "guardrails": {
        "yaml": "...yaml_config_content..",
        "colang": "...colang_config_content.."
    }
}
```