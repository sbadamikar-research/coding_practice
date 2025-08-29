import math
import numpy as np
import pandas as pd
from collections import Counter

def compute_entropy(target: pd.DataFrame) -> float:
    entropy = 0
    for count in target.value_counts().values:
        class_probablity = (count / target.value_counts().sum()) 
        entropy += (class_probablity * math.log(class_probablity, 2))
    
    return -1 * entropy        

def compute_information_gain(df: pd.DataFrame, splitting_attr: str, target_attr: str, data_entropy: float) -> float:
    
    subsets = dict(tuple(df.groupby(splitting_attr)))
    subset_entropies = [ (subsets[key].shape[0] / df.shape[0]) * compute_entropy(subsets[key][target_attr]) 
                        for key in subsets.keys()]
    return (((data_entropy - np.array(subset_entropies)).sum()), subsets)
    
    
def learn_decision_tree(examples: list[dict], attributes: list[str], target_attr: str) -> dict:
 
    df = pd.DataFrame(examples)
    print(df.head())
    print(attributes)
    
    # If all the values in the target are the same, that is the answer
    if(df[target_attr].value_counts().size == 1):
        print("Leaf: ", df[target_attr].values[0])
        return df[target_attr].values[0]
    
    # Guaranteed leaf node because no more attributes to split, return mode
    if not attributes:
        return Counter(df[target_attr]).most_common(1)[0][0]
    
    data_entropy = compute_entropy(df[target_attr])
    max_gain = 0
    best_attr = 0
    best_subsets = dict()
    for attr in attributes:
        IG_attr, subsets = compute_information_gain(df, attr, target_attr, data_entropy)
        print(subsets)
        if(IG_attr > max_gain):
            max_gain = IG_attr
            best_attr = attr
            best_subsets = subsets
    
    subtree_root_node = {best_attr : {}}
    
    new_attr = [a for a in attributes if a != best_attr]
    for val in best_subsets.keys():
        print("- ", best_subsets[val])
        subtree_root_node[best_attr][val] = learn_decision_tree(best_subsets[val].to_dict('records'), new_attr, target_attr)
        
    return subtree_root_node


examples = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Strong', 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'},
    {'Outlook': 'Rain', 'Temperature': 'Mild', 'Humidity': 'High', 'Wind': 'Weak', 'PlayTennis': 'Yes'}
]
attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
# print(learn_decision_tree(examples, attributes, 'PlayTennis'))

print(learn_decision_tree([ 
                           {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'}, 
                           {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'}, 
                           {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}, 
                           {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'}, 
                           {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'Yes'}, 
                           {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'}, 
                           {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'}, 
                           {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'} ], ['Outlook', 'Wind'], 'PlayTennis'))

# {
#     'Outlook': {
#         'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}},
#         'Overcast': 'Yes',
#         'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}
#     }
# }

# {'Outlook': {'Sunny': {'Wind': {'Weak': 'No', 'Strong': 'No'}}, 'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}, 'Overcast': 'Yes'}}
