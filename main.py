import os
from config import FEATURE_PATH
import torch
# dataset
from dataset.get_feature_data import get_feature_dataset
from dataset.get_standard_rawdata import get_standard_rawdataset
from dataset.get_test_data import get_test_dataset
# Classical model
from models.DNN import DNN
from models.CNN import CNN
from models.StackLSTM import StackLSTM
from models.BiLSTMAttention import BiLSTMAttention
# Deep model
from models.DCNN import DCNN
# Pretrain & fusion model
from models.DNN_fusion import DNN as DNN_fusion
from models.DCNN_fusion import DCNN as DCNN_fusion
from models.Fusion import Fusionnet
from feature_extraction.get_fusion_feature import fusion_feature
# from dataset.get_fusion3 import get_fusion_dataset
# Trainer
from trainer.NormTrainer import NormTrainer
from trainer.PreTrainer import PreTrainer
# Tester
from tester.Tester import Tester
# Predicter
from tester.Predicter import Predicter
# Evaluation
from utils.evaluation import eval_func


Classical_train_PATH = os.path.join(FEATURE_PATH, "Bern_classical_features_20s.csv")

Deep_train_PATH = os.path.abspath(os.path.join(__file__, "../data/Standard Bern Barcelona/"))
Deep_test_PATH = os.path.abspath(os.path.join(__file__, "../data/Standard Bern Barcelona/"))
Fusion_PATH = "D:/GitHub/EZ-identify/data/new_feature/bern_fusion_feature.csv"



def Classical_train():
    model = DNN()                  # 实例化模型
    # model = DCNN()
    # model = StackLSTM()
    # model = BiLSTMAttention()
    traindataset = get_feature_dataset(Classical_train_PATH)     # 得到训练数据集
    testdataset = get_feature_dataset(Classical_test_PATH)     # 得到测试数据集
    criterion = torch.nn.BCELoss()    # 损失函数
    trainer = PreTrainer(model, traindataset, testdataset,criterion, eval_func, n_epoch=100, n_batch=20)
    trainer.train()
    trainer.test()

def Deep_train():
    model = DCNN()
    traindataset = get_standard_rawdataset(Deep_train_PATH,in_mem=True)     # 得到训练数据集
    testdataset = get_standard_rawdataset(Deep_test_PATH,in_mem=True)     # 得到测试数据集
    print("data load ok!")
    criterion = torch.nn.BCELoss()
    trainer = PreTrainer(model, traindataset, testdataset, criterion, eval_func, n_epoch=100, n_batch=20)
    trainer.train()
    trainer.test()

def get_fusion_feature():
    # 融合特征提取
    modela = DNN_fusion()
    modelb = DCNN_fusion()
    modela.load("./checkpoints/DNN_12_10_11_30_23.pth")
    modelb.load("./checkpoints/DCNN_12_15_22_57_19.pth")
    # 融合分类器的训练集
    dataseta = get_feature_dataset(Classical_train_PATH)
    datasetb = get_standard_rawdataset(Deep_train_PATH,in_mem=True)
    fusion_train = fusion_feature(modela,modelb,dataseta,datasetb)
    fusion_train.get_feature_csv() # 保存成csv
    # 融合分类器的测试集
    datasetc = get_feature_dataset(Classical_test_PATH)
    datasetd = get_standard_rawdataset(Deep_test_PATH,in_mem=True)
    fusion_test = fusion_feature(modela,modelb,datasetc,datasetd)
    fusion_test.get_feature_csv() # 保存成csv

def finnal_fusion():
    # 融合特征进浅层模型决策
    model = Fusionnet()
    traindataset = get_feature_dataset(Fusion_train_PATH)
    testdataset = get_feature_dataset(Fusion_test_PATH)
    criterion = torch.nn.BCELoss()
    trainer = PreTrainer(model, traindataset, testdataset ,criterion, eval_func, n_epoch=50, n_batch=20)
    trainer.train()
    trainer.test()

def predict():
    model = DNN()
    model.load("./checkpoints/DNN_12_10_11_30_23.pth")
    test_dataset = get_feature_dataset(PATH)
    tester = Tester(model, test_dataset,eval_func)
    tester.test()


def main():
 # 训练经典分支模型，并选择和固定模型参数；使用训练集和验证集
 #    Classical_train()
 # 训练深度分支模型，并选择和固定模型参数；使用训练集和验证集
 #    Deep_train()
 # 融合特征，并选择和固定浅层分类器；使用训练集和验证集
 #    get_fusion_feature()
 # 使用测试集，测试融合模型
    finnal_fusion()


if __name__ == "__main__":
    main()