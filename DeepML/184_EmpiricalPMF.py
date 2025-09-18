from collections import Counter
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    
    counts = Counter(samples)
    pmf = []
    for item in counts.items():
        pmf.append([item[0], round((item[1]/counts.total()),4)])
    
    return pmf
    
samples = [1, 2, 2, 3, 3, 3]
print(empirical_pmf(samples))