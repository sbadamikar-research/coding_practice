import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    kernel_height, kernel_width = kernel.shape
    padded_input = np.pad(input_matrix, padding, constant_values=(0,0))
    padded_height, padded_width = padded_input.shape
    
    output_set = []
    y = 0
    while (y + kernel_height - 1 < padded_height):
        conv_set=[]
        x = 0
        while (x + kernel_width -1 < padded_width):
            conv = np.sum(padded_input[y:y+kernel_height, x:x+kernel_width] * kernel)
            conv_set.append(conv)
            x = x + stride
        
        output_set.append(conv_set)
        y = y + stride

    output_matrix = np.array(output_set)
        
    return output_matrix


############################################
###               TESTING                ###
############################################

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)