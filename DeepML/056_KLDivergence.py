import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	
    log_term = np.log(sigma_q / sigma_p)
    num = (sigma_p ** 2) + ( (mu_p - mu_q) ** 2)
    denom = 2 * (sigma_q ** 2)
    
    return (log_term + (num/denom) - 0.5)
