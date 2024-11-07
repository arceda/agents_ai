from llm_sandbox import SandboxSession

# Create a new sandbox session
# it pull a docker image
with SandboxSession(image="python:3.9.19-bullseye", keep_template=True, lang="python") as session:
    result = session.run("print('Hello, World!')")
    print(result)