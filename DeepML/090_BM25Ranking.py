import numpy as np
from collections import Counter

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):

	N = len(corpus)	
	scores = np.zeros(shape=(N,), dtype="float")
	term_freq = np.zeros(shape=(N,), dtype="float")

	doc_len = []
	doc_counts = []
	for i, doc in enumerate(corpus):
		doc_counts.append(Counter(doc))
		doc_len.append(len(doc))
	doc_len = np.array(doc_len)
	
	for term in query:
		# Get term frequencies
		for i, counts in enumerate(doc_counts):
			term_freq[i] = counts[term]

		doc_freq = (term_freq > 0).sum()

		inv_doc_freq = np.log( (N + 1) / (doc_freq + 1))

		norm_len = 1 - b + (b * doc_len / doc_len.mean())
		adjusted_TF = (term_freq * (k1 + 1)) / (term_freq + (k1 * norm_len))

		term_score = inv_doc_freq * adjusted_TF
	
		scores += term_score

	return np.round(scores, 3)

corpus = [['the', 'cat', 'sat'], ['the', 'dog', 'ran'], ['the', 'bird', 'flew']]
query = ['the', 'cat']
print(calculate_bm25_scores(corpus, query))

print(calculate_bm25_scores([['term'] * 10, ['the'] * 2], ['term'], k1=1.0))
# Result: [0.705, 0]