import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
import numpy as np
import torch
import pickle
import torch.utils.data as Data
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.nn import BCEWithLogitsLoss, BCELoss, MultiLabelSoftMarginLoss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2
import scipy.stats as st
import sys
sys.path.append(r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验")
from utils import array_to_tensor, Data_set, model, hamming_score, predict
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import seaborn as sns

gas_model = model()
weights_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\模型保存\model3.pt"
gas_model.load_state_dict(torch.load(weights_save_path, map_location=torch.device('cpu')))
gas_model.eval()


path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\测试集\test_data.npy"
path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\测试集\test_label.npy"
test_input = np.load(path1)
test_label = np.load(path2)

inputs = torch.tensor(test_input, dtype=torch.float32)
print(inputs.shape)
labels = torch.tensor(test_label, dtype=torch.float32)
print(labels.shape)
print(labels.dtype)

with torch.no_grad():
    logits = gas_model(inputs)  # logits = torch.Size([1015, 6])
    logits[:, :3] = torch.where(logits[:, :3]>0.5, 1, 0)
    components = ["methan", "acetone", "water vapor"]
    predictions = logits[:, :3].detach().cpu().numpy()
    labels = test_label[:, :3]
    print(predictions.shape)
    print(labels.shape)


    def transform_labels(old_):
        new_ = np.zeros((old_.shape[0],))
        for i in range(old_.shape[0]):
            if (old_[i, :] == np.array([1, 1, 1])).all():
                new_[i] = 1.0
            elif (old_[i, :] == np.array([1, 1, 0])).all():
                new_[i] = 2.0
            elif (old_[i, :] == np.array([1, 0, 1])).all():
                new_[i] = 3.0
            elif (old_[i, :] == np.array([0, 1, 1])).all():
                new_[i] = 4.0
            elif (old_[i, :] == np.array([1, 0, 0])).all():
                new_[i] = 5.0
            elif (old_[i, :] == np.array([0, 1, 0])).all():
                new_[i] = 6.0
            elif (old_[i, :] == np.array([0, 0, 1])).all():
                new_[i] = 7.0
        return new_


    new_pred = transform_labels(predictions)
    new_label = transform_labels(labels)

    cm = confusion_matrix(new_label, new_pred)

    xtick = ["methane acetone water", "methane acetone", "methane water", "acetone water",
             "methane", "acetone", "water"]
    ytick = ["methane acetone water", "methane acetone", "methane water", "acetone water",
             "methane", "acetone", "water"]

    plt.figure()
    sns.heatmap(cm, fmt='g', annot=True, cbar=True, xticklabels=xtick, yticklabels=ytick)
    plt.show()