import numpy as np
def translate_object(points: list[list], tx: float, ty: float):

    for i in range(len(points)):
        points[i][0] += tx
        points[i][1] += ty
        
    return points


### TESTING

points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points, tx, ty))