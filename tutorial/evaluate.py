import dspy
from tutorial.data import test_set, train_set
from tutorial.signature import people_extractor

def extraction_correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """
    Computes correctness of entity extraction predictions.
    
    Args:
        example (dspy.Example): The dataset example containing expected people entities.
        prediction (dspy.Prediction): The prediction from the DSPy people extraction program.
        trace: Optional trace object for debugging.
    
    Returns:
        bool: True if predictions match expectations, False otherwise.
    """
    return prediction.extracted_people == example.expected_extracted_people

evaluate_correctness = dspy.Evaluate(
    devset=test_set,
    metric=extraction_correctness_metric,
    num_threads=24,
    display_progress=True,
    display_table=True
)

lm = dspy.LM('ollama_chat/qwen2.5-coder', api_base='http://localhost:11434', api_key='')
dspy.settings.configure(lm=lm)


mipro_optimizer = dspy.MIPROv2(
    metric=extraction_correctness_metric,
    auto="medium",
)
optimized_people_extractor = mipro_optimizer.compile(
    people_extractor,
    trainset=train_set,
    max_bootstrapped_demos=4,
    requires_permission_to_run=False,
    minibatch=False
)

optimized_people_extractor.save("optimized_extractor.json")

evaluate_correctness(people_extractor, devset=test_set)