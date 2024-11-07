from nemoguardrails import RailsConfig, LLMRails

config = RailsConfig.from_path("./7_config_topical_rails")
rails = LLMRails(config)

response = rails.generate(messages=[{
    "role": "user",
    "content": "How can I cook an apple pie?"
}])
print(response["content"])