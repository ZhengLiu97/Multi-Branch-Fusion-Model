from sklearn.metrics import accuracy_score, confusion_matrix

def get_acc(target, predict, threshold=0.5):
    predict_label = (predict >= threshold).astype(int)
    target_label = (target >= 0.5).astype(int)
    return accuracy_score(target_label,predict_label)

def get_eva(target, predict, threshold=0.5):
    predict_label = (predict >= threshold).astype(int)
    target_label = (target >= 0.5).astype(int)
    cm = confusion_matrix(target_label, predict_label)
    TN, FP, FN, TP = cm.ravel()
    ACC = (TP+TN)/(TP+FP+FN+TN)  # ACC：classification accuracy，描述分类器的分类准确率
    TPR = TP / (TP + FN)  # 敏感度（sensitivity）true positive rate，描述识别出的所有正例占所有正例的比例
    FPR = FP / (FP + TN)  # false positive rate，描述将负例识别为正例的情况占所有负例的比例
    TNR = TN / (FP + TN)  # 特异度（specificity） true negative rate，描述识别出的负例占所有负例的比例
    PPV = TP / (TP + FP)
    NPV = TN / (FN + TN)
    return ACC, TPR, FPR, TNR, PPV, NPV,TN, FP, FN, TP

def eval_func(target, predict):
    # acc = get_acc(target, predict)
    ACC, TPR, FPR, TNR, PPV, NPV,TN, FP, FN, TP = get_eva(target, predict)
    return ACC, TPR, FPR, TNR, PPV, NPV ,TN, FP, FN, TP
