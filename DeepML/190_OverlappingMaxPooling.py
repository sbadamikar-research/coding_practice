import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).
    Uses ceil mode for output dimensions (allows partial windows at boundaries).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int), must be < kernel_size

    Returns:
        A 4D tensor after overlapping pooling with ceil mode.
    """
    
    pool_h = np.ceil((x.shape[2] - kernel_size) / stride).astype(int) + 1
    pool_w = np.ceil((x.shape[3] - kernel_size) / stride).astype(int) + 1
    max_pool = np.zeros((x.shape[0], x.shape[1], pool_h, pool_w))
    # print(max_pool.shape)

    if (stride > kernel_size):
        return max_pool

    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for i in range(pool_h):
                for j in range(pool_w):
                    h = i * stride
                    w = j * stride
                    h_end = min(x.shape[2], h + kernel_size)
                    w_end = min(x.shape[3], w + kernel_size)
                    kernel = x[n, c, h:h_end, w:w_end]
                    max_pool[n, c, i, j] = np.max(kernel)
    
    return max_pool

    

### TESTING
# x = np.arange(1, 17).reshape(1, 1, 4, 4)
# print(overlapping_max_pool2d(x, kernel_size=3, stride=2))

# np.random.seed(0) 
# x = np.random.randn(1, 2, 5, 5)
x = np.arange(1, 10).reshape(1, 1, 3, 3)
print(x)
print(overlapping_max_pool2d(x, 2, 1))