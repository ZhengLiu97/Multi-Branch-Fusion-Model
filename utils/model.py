import os
import torch
from config import DATA_ROOT_PATH

def get_sigmoid_label(predict, threshold=0.5):
    predict_label = (predict >= threshold).astype(int)
    return predict_label

def predict(model, input):
    model.eval()
    with torch.no_grad():
        pred = model(input)
    return pred