def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

    if (not (len(a))):
        return -1

	if (len(a[0]) != len(b)):
        return -1

    retval = []

    for row in a:
        sum = 0
        for i, col in enumerate(row):
            sum += col * b[i]
        retval.append(sum)

    return retval


    