import dspy

lm = dspy.LM('ollama_chat/qwen2.5-coder', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)