def calculate_brightness(img):
    
    # Check datatype;
    if not isinstance(img, list):
        return -1

    # Empty image
    if not len(img):
        return -1
    
    height = len(img)
    width = None
    total = 0

    for row in img:
        if not isinstance(row, list):
            return -1
        
        if isinstance(width, int):
            if (len(row) != width):
                return -1
        else:
            width = len(row)
        
        for pixel_val in row:
            if pixel_val < 0:
                return -1
            
            if pixel_val > 255:
                return -1
            
            total += pixel_val 

    return (total / (height * width))

### TESTING

img = [
    [100, 200],
    [50, 150]
]
print(calculate_brightness(img))