# References:
# https://www.statlect.com/matrix-algebra/change-of-basis

import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B = np.array(B)
    C = np.array(C)
    
    # For:
    #   v = C * v_c         (1)
    #   v = B * v_b         (2)
    #   v_c = P * v_b       (3)
    #
    # (1) + (2) => C * v_c = B * v_b
    #              v_c = C_inv * B * v_b    (4)
    # (3) + (4) =? P = C_inv * B
    P = np.round(np.linalg.inv(C) @ B, 4)
    return P

print(transform_basis([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]]))
# [[-0.6772, -0.0126, 0.2342], [-0.0184, 0.0505, -0.0275], [0.5732, -0.0345, -0.0569]]

print(transform_basis([[1,0],[0,1]],[[1,2],[9,2]]))
# [[-0.125, 0.125 ],[ 0.5625, -0.0625]]