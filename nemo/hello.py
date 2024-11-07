from nemoguardrails import RailsConfig
from nemoguardrails import LLMRails
from dotenv import load_dotenv, find_dotenv
load_dotenv()

config = RailsConfig.from_path("./config")
rails = LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": "Hello!"
}])
print(response["content"])

