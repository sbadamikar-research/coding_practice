def confusion_matrix(data):
	
    confusion_matrix = [[0, 0], [0, 0]]

    for y_true, y_pred in data:
        confusion_matrix[not y_true][not y_pred] += 1

    return confusion_matrix

### TESTING 

data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
print(confusion_matrix(data))