# author: SaKuRa Pop
# data: 2021/12/1 12:42
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import pickle
import torch.utils.data as Data
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.nn import BCEWithLogitsLoss, BCELoss, MultiLabelSoftMarginLoss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import hamming_loss
import wandb
from sklearn.metrics import accuracy_score


def array_to_tensor(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.type(torch.cuda.FloatTensor)
    return tensor


def Data_set(x, y, batch_size):
    """
    生成data_loader实例。可以定义batch_size
    :param input_data: 希望作为训练input的数据，tensor类型
    :param label_data: 希望作为训练label的数据，tensor类型
    :param batch_size: batch size
    :return: data_loader实例
    """
    data_set = Data.TensorDataset(x, y)
    data_loader = Data.DataLoader(dataset=data_set,
                                  shuffle=True,
                                  batch_size=batch_size,
                                  num_workers=0)
    return data_loader


class model(nn.Module):
    """
    神经网络滤波器
    input -     Conv1 -      Maxpool -  Conv2 -      Maxpool -    Conv3 -     Maxpool  -  Conv4 -     Conv5 - Globaverg
    (1, 2000) - (10, 997) -  (10, 498)- (100, 248) - (100, 122) - (500, 60) - (500, 29) - (1000, 14)-(2000, 6)-(2000,1)
    """

    def __init__(self, out_features1, out_features2, out_features3, out_features4, out_features5):
        super(model, self).__init__()
        self.fc1 = nn.Linear(in_features=3321, out_features=out_features1)
        nn.init.xavier_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(num_features=out_features1)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')

        self.fc2 = nn.Linear(in_features=out_features1, out_features=out_features2)
        nn.init.xavier_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(num_features=out_features2)
        # nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

        # self.dropout1 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(in_features=out_features2, out_features=out_features3)
        nn.init.xavier_normal_(self.fc3.weight)
        self.bn3 = nn.BatchNorm1d(num_features=out_features3)

        self.fc4 = nn.Linear(in_features=out_features3, out_features=out_features4)
        nn.init.xavier_normal_(self.fc4.weight)
        self.bn4 = nn.BatchNorm1d(num_features=out_features4)

        self.fc5 = nn.Linear(in_features=out_features4, out_features=out_features5)
        nn.init.xavier_normal_(self.fc5.weight)
        self.bn5 = nn.BatchNorm1d(num_features=out_features5)

        self.fc6 = nn.Linear(in_features=out_features5, out_features=6)
        nn.init.xavier_normal_(self.fc6.weight)
        self.bn6 = nn.BatchNorm1d(num_features=6)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.sigmoid(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.sigmoid(x)

        # x = self.dropout1(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x_1 = torch.sigmoid(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x_2 = torch.sigmoid(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x_3 = torch.sigmoid(x)

        x = self.fc6(x_3)
        x = self.bn6(x)
        x = torch.sigmoid(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)  # normal: mean=0, std=1
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')


def train_batch_loss(model, loss_function1, loss_function2, x, y, opt):
    pred = model(x)
    loss1 = loss_function1(pred[:, :3], y[:, :3])
    loss2 = loss_function2(pred[:, 3:], y[:, 3:])
    loss = loss1 + loss2
    loss.backward()
    opt.step()
    opt.zero_grad()
    return np.array(loss.cpu().detach().numpy()), np.array(len(x))


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def valid_batch_loss(model, loss_function1, loss_function2, x, y):
    pred = model(x)
    loss1 = loss_function1(pred[:, :3], y[:, :3])
    loss2 = loss_function2(pred[:, 3:], y[:, 3:])
    loss = loss1 + loss2
    prediction = np.zeros_like(pred.cpu().detach().numpy()[:, :3])
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if pred[i, j] > 0.5:
                prediction[i, j] = 1
            else:
                prediction[i, j] = 0
    accuracy = accuracy_score(y.cpu().detach().numpy()[:, :3],
                              prediction)
    hamming = hamming_score(y[:, :3].cpu().detach().numpy(),
                            prediction)
    return np.array(loss.cpu().detach().numpy()), accuracy, hamming, np.array(len(x))


def training(train_x, train_y, val_x, val_y, Model, epoch, batch_size,
             learning_rate, out_features1, out_features2, out_features3, out_features4,
             out_features5):
    train_data_set = Data.TensorDataset(train_x, train_y)
    train_data_set = Data.DataLoader(dataset=train_data_set,
                                     shuffle=True,
                                     batch_size=batch_size,
                                     num_workers=0)

    val_data_set = Data.TensorDataset(val_x, val_y)
    val_data_set = Data.DataLoader(dataset=val_data_set,
                                   shuffle=True,
                                   batch_size=batch_size,
                                   num_workers=0)

    gas_model = Model(out_features1, out_features2, out_features3, out_features4, out_features5).to(Gpu)
    optimizer = torch.optim.Adam(gas_model.parameters(), lr=learning_rate)

    for e in range(epoch):
        gas_model.train()
        loss, number_of_data = zip(*[train_batch_loss(gas_model, criterion1, criterion2, x, y, optimizer)
                                     for x, y in train_data_set])
        train_loss = np.sum(np.multiply(loss, number_of_data)) / np.sum(number_of_data)

        gas_model.eval()
        with torch.no_grad():
            val_loss, val_accuracy, val_hamming_score, number_of_data = zip(*[valid_batch_loss(gas_model,
                                                                                               criterion1,
                                                                                               criterion2, x, y)
                                                                              for x, y in val_data_set])
            val_loss = np.sum(np.multiply(val_loss, number_of_data)) / np.sum(number_of_data)
            val_accuracy = np.mean(np.array(val_accuracy))

        print("[Epoch {}/{} Train loss:{:.6f}]\t Validation loss:{:.6f}\t Validation accuracy:{:.3f}%".format(e + 1,
                                                                                                              epoch,
                                                                                                              train_loss,
                                                                                                              val_loss,
                                                                                                              val_accuracy))
        wandb.log()

if __name__ == "__main__":

    path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\训练集+验证集\train_data.npy"
    path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\训练集+验证集\train_label.npy"
    train_input = np.load(path1)
    train_label = np.load(path2)

    path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\训练集+验证集\val_data.npy"
    path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\训练集+验证集\val_label.npy"
    valid_input = np.load(path1)
    valid_label = np.load(path2)

    train_input = array_to_tensor(train_input)
    train_label = array_to_tensor(train_label)
    valid_input = array_to_tensor(valid_input)
    valid_label = array_to_tensor(valid_label)

    Gpu = torch.device("cuda")

    criterion1 = BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()

    training(train_input, train_label, valid_input,
             valid_label, model, 500,
             128, 0.0001,
             1000, 1000,
             1000, 1000,
             500)
