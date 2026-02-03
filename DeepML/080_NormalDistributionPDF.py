import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	exponent = -0.5 * ( ((x - mean) / std_dev) ** 2)
	val = math.exp(exponent) / (std_dev * math.sqrt(2 * math.pi))
	return round(val, 5)


### TESTING

x = 16
mean = 15
std_dev = 2.04
print(normal_pdf(x, mean, std_dev))