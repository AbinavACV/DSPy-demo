from typing import List
import dspy

class PeopleExtraction(dspy.Signature):
    """
    Extract contiguous tokens referring to specific people, if any, from a list of string tokens.
    Output a list of tokens. In other words, do not combine multiple tokens into a single value.
    """
    tokens: list[str] = dspy.InputField(desc="tokenized text")
    extracted_people: list[str] = dspy.OutputField(desc="all tokens referring to specific people extracted from the tokenized text")

people_extractor = dspy.ChainOfThought(PeopleExtraction) 