import numpy as np
from collections import Counter

def pass_at_1(responses_correct: np.ndarray) -> float:
    """
    Compute pass@1 by averaging correctness.

    Args:
        responses_correct: Boolean array for each response
        
    Returns:
        pass@1 score
    """

    return (responses_correct.sum() / responses_correct.size)


def majority_voting(responses: list[str]) -> str:
    """
    Return the most common response.

    Args:
        responses: List of response strings
        
    Returns:
        Most frequent response
    """
    
    return Counter(responses).most_common(1)[0][1]


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute unbiased pass@k from n samples with c correct.

    Formula: pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: Total samples
        c: Correct samples
        k: k in pass@k
        
    Returns:
        Estimated pass@k
    """
    probability = 1;

    for i in range(k):
        probability *= ((n-c-i) / (n-i))
    return 1 - probability
                

### TESTING

responses_correct = np.array([True, False, True, False])
print(f"Pass@1: \n {pass_at_1(responses_correct)}")
print(f"Majority Voting: \n {majority_voting(responses_correct)}")