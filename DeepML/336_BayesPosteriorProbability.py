import numpy as np
def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """
    Calculate posterior probabilities using Bayes' Theorem.

    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis
        
    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """
    
    # P (A|B) = P(B|A) * P(A) / P(B)

    priors = np.array(priors)
    likelihoods = np.array(likelihoods)

    return (priors * likelihoods) / (priors * likelihoods).sum()