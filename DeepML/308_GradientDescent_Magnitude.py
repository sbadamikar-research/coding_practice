import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.

    Args:
            gradient: A list representing the gradient vector

    Returns:
            Dictionary containing:
            - magnitude: The L2 norm of the gradient
            - direction: Unit vector in direction of steepest ascent
            - descent_direction: Unit vector in direction of steepest descent
    """
    gradient = np.array(gradient)

    mag = np.sqrt(gradient @ gradient)

    if (mag == 0):
        return {
            'magnitude': 0.0,
            'direction': (np.zeros_like(gradient)).tolist(),
            'descent_direction': (np.zeros_like(gradient)).tolist()
        }


    return {
        'magnitude': float(mag),
        'direction': (gradient / mag).tolist(),
        'descent_direction': (-1 * gradient / mag).tolist()
    }

print(gradient_direction_magnitude([3.0, 4.0]))
print(gradient_direction_magnitude([0.0, 0.0]))