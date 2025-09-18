import math

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    # For a matrix M [[a , b], [c, d]], s.t. l = lambda
    #
    # Since det(A-lI) = 0
    # (a-l)(d-l) - bc = 0
    # l^2 - (a+d)l + (ad-bc) = 0
    # Using form A*l^2 + Bl + C = 0
    
    A = 1
    B = matrix[0][0] + matrix[1][1]
    C = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    
    # Solutions to the quadratic are: 
    # 1. -B + sqrt(B^2 - 4AC) / 2A
    # 2. -B - sqrt(B^2 - 4AC) / 2A
    
    square =  math.pow(B,2) - (4*A*C)
    if (square < 0):
        raise Exception("No valid solutions to quadratic")
    
    retval = [abs((-B + math.sqrt(square)) / (2*A)),
              abs((-B - math.sqrt(square)) / (2*A))]
    retval.sort(reverse=True)
    
    return retval