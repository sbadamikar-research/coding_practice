import numpy as np

def deterministic_hash(s):
    '''Converts a string to a deterministic integer.'''
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h

def create_hv(dim, seed):
    '''Creates a bipolar hypervector of given dimension using the seed.'''
    np.random.seed(seed % (2**32 - 1))
    return np.random.choice([-1, 1], dim)

def create_row_hv(row: dict, dim: int, random_seeds: dict):
    '''Create composite hypervector for a dataset row.
    
    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.
    '''

    row_hv = create_hv(dim=dim, seed=42) * 0

    for feature_name, feature_value in row.items():
        name_seed = random_seeds[feature_name]
        value_seed = random_seeds[feature_name] + deterministic_hash(feature_value)

        name_hv = create_hv(dim=dim, seed=name_seed)
        value_hv = create_hv(dim=dim, seed=value_seed)

        feature_hv = name_hv * value_hv
        row_hv += feature_hv
    
    return np.where(row_hv >= 0, 1, -1)

### TESTING

row = {'FeatureA': 'value1', 'FeatureB': 'value2'}
dim = 5
random_seeds = {'FeatureA': 42, 'FeatureB': 7}
print(create_row_hv(row, dim, random_seeds))