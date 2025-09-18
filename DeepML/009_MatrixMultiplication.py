import numpy as np
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
	try:
        retval = np.array(a) @ np.array(b)
    except:
        retval = -1

    return retval