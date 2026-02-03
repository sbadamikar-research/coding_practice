import numpy as np
from collections import Counter


def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.

    Args:
        data: List or numpy array of numerical values

    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    
    data.sort()
    counts = Counter(data)
    total_counts = len(data)
    middle_value = ((total_counts + 1) / 2)

    total = 0
    for val in data:
        total += val

    data_mean = total/total_counts
    data_median = 0.5 * (data[np.floor(middle_value-1).astype(int)] +
                         data[np.ceil(middle_value-1).astype(int)])

    lower_mid = (np.floor(middle_value) + 1) / 2
    lower_quartile = 0.5 * (data[np.floor(lower_mid-1).astype(int)] +
                            data[np.ceil(lower_mid-1).astype(int)])

    upper_mid = (np.ceil(middle_value) - 1) + lower_mid
    upper_quartile = 0.5 * (data[np.floor(upper_mid-1).astype(int)] +
                            data[np.ceil(upper_mid-1).astype(int)])

    errors = []
    total_error_sq = 0
    for val in data:
        error = val - data_mean
        errors.append(error)
        total_error_sq += (error ** 2)
    variance = total_error_sq / total_counts
    std_dev = np.round(np.sqrt(variance), 4).astype(float)

    return {"mean": data_mean,
            "median": data_median,
            "mode": counts.most_common(1)[0][0],
            "variance": variance,
            "standard_deviation": std_dev,
            "25th_percentile": lower_quartile,
            "50th_percentile": data_median,
            "75th_percentile": upper_quartile,
            "interquartile_range": (upper_quartile - lower_quartile)
            }


# TESTING

print(descriptive_statistics([1, 2, 2, 3, 4, 4, 4, 5]))
print(descriptive_statistics([10, 20, 20, 30, 40]))
