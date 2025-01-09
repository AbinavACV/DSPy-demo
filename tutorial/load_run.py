import dspy
from tutorial.signature import PeopleExtraction

lm = dspy.LM('ollama_chat/qwen2.5-coder', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)


loaded_people_extractor = dspy.ChainOfThought(PeopleExtraction)
loaded_people_extractor.load("optimized_extractor.json")

extracted = loaded_people_extractor(tokens=["Canada", "recalled", "Justin", "Trudeau"]).extracted_people

print(extracted)