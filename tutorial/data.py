import os
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List
import dspy

def load_conll_dataset() -> dict:
    """
    Loads the CoNLL-2003 dataset into train, validation, and test splits.
    
    Returns:
        dict: Dataset splits with keys 'train', 'validation', and 'test'.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a temporary Hugging Face cache directory for compatibility with certain hosted notebook
        # environments that don't support the default Hugging Face cache directory
        os.environ["HF_DATASETS_CACHE"] = temp_dir
        return load_dataset("conll2003", trust_remote_code=True)

def extract_people_entities(data_row: Dict[str, Any]) -> List[str]:
    """
    Extracts entities referring to people from a row of the CoNLL-2003 dataset.
    
    Args:
        data_row (Dict[str, Any]): A row from the dataset containing tokens and NER tags.
    
    Returns:
        List[str]: List of tokens tagged as people.
    """
    return [
        token
        for token, ner_tag in zip(data_row["tokens"], data_row["ner_tags"])
        if ner_tag in (1, 2)  # CoNLL entity codes 1 and 2 refer to people
    ]

def prepare_dataset(data_split, start: int, end: int) -> List[dspy.Example]:
    """
    Prepares a sliced dataset split for use with DSPy.
    
    Args:
        data_split: The dataset split (e.g., train or test).
        start (int): Starting index of the slice.
        end (int): Ending index of the slice.
    
    Returns:
        List[dspy.Example]: List of DSPy Examples with tokens and expected labels.
    """
    return [
        dspy.Example(
            tokens=row["tokens"],
            expected_extracted_people=extract_people_entities(row)
        ).with_inputs("tokens")
        for row in data_split.select(range(start, end))
    ]

# Load the dataset
dataset = load_conll_dataset()

# Prepare the training and test sets
train_set = prepare_dataset(dataset["train"], 0, 50)
test_set = prepare_dataset(dataset["test"], 0, 200)
