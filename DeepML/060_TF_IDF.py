import numpy as np
from collections import Counter

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """

    tf_idf = []
    doc_counts = [Counter(doc) for doc in corpus]
    for q in query:
        freqs = np.array([ (counts[q] / counts.total()) for counts in doc_counts])

        df = (freqs > 0).sum()
        id_freq = np.log( (len(corpus) + 1) / (df + 1) ) + 1

        tf_idf.append(freqs * id_freq)

    return np.array(tf_idf).T.tolist()



### TESTING

corpus = [ ["the", "cat", "sat", "on", "the", "mat"], ["the", "dog", "chased", "the", "cat"], ["the", "bird", "flew", "over", "the", "mat"] ] 
query = ["cat", "mat"]

print(compute_tf_idf(corpus, query))