def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	
    TP = FP = FN = TN = 0

    for y_true, y_pred in zip(actual, predicted):
        TP += (y_true and y_pred)
        FP += ((not y_true) and y_pred)
        TN += ((not y_true) and (not y_pred))
        FN += ((y_true) and (not y_pred))
    
    count = TP + FP + FN + TN
    confusion_matrix = [[TP, FN], [FP, TN]]

    accuracy = (TP + TN) / count

    precision = TP / (TP + FP)
    negativePredictive = TN / (TN + FN)
    
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    f1 = 2 * (precision * recall) / (precision + recall)

    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
