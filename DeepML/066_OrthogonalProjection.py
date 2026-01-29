def dot(v1, v2):
    ans = 0

    for i in range (len(v1)):
        ans += (v1[i] * v2[i])
    
    return ans

def orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """
    ans = []
    scale = (dot(v, L) / dot(L, L))
    for i in range(len(L)):
        ans.append(scale * L[i])

    return ans


def np_orthogonal_projection(v, L):
    """
    Compute the orthogonal projection of vector v onto line L.

    :param v: The vector to be projected
    :param L: The line vector defining the direction of projection
    :return: List representing the projection of v onto L
    """

    import numpy as np
    
    v = np.array(v)
    L = np.array(L)
    return ( ((v @ L.T) / (L @ L.T)) * L )

### TESTING

v = [3, 4]
L = [1, 0]
print(np_orthogonal_projection(v, L))