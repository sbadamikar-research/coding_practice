import numpy as np

def pos_encoding(position: int, d_model: int):
	
	pos_encoding = np.ones((position, d_model), dtype="float16")
	print(pos_encoding.shape)

	for pos in range(position):
		for i in range(int(d_model/2)):
			angle = pos / (10000 ** ((2 * i) / d_model))
			pos_encoding[pos, (2 * i)] = np.sin(angle)
			pos_encoding[pos, ((2 * i) + 1)] = np.cos(angle)

	return np.float16(pos_encoding)

print(pos_encoding(2, 8))

# print(pos_encoding(5, 16))
