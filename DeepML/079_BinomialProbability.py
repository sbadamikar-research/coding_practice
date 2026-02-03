import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes
    """
    if (k > n):
        return 0.0
    
    nCk = math.factorial(n) / (math.factorial(k) * math.factorial(n-k))

    return (nCk * (p ** k) * ((1-p) ** (n-k)))

print(round(binomial_probability(6, 4, 0.7), 10))