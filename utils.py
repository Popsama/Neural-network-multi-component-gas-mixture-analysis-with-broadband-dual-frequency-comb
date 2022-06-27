# author: SaKuRa Pop
# data: 2021/12/15 15:17
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np
import torch
import torch.utils.data as Data


def array_to_tensor(array):
    tensor = torch.from_numpy(array)
    tensor = tensor.type(torch.cuda.FloatTensor)
    return tensor


def relative_error(true, predict):
    error = (np.abs(true-predict)/true)
    return error


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

    def __init__(self):
        super(model, self).__init__()
        self.fc1 = nn.Linear(in_features=3321, out_features=196)
        nn.init.xavier_normal_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(num_features=196)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(in_features=196, out_features=774)
        nn.init.xavier_normal_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(num_features=774)
        # nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        # self.dropout1 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(in_features=774, out_features=211)
        nn.init.xavier_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(in_features=211, out_features=6)
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = torch.sigmoid(x)
        # x = self.dropout1(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x

    def initialize(self):
        for m in self.modules():
            # 判断这一层是否为线性层，如果为线性层则初始化权值
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)  # normal: mean=0, std=1
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')


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


def predict(logits, labs):
    components = ["甲烷", "丙酮", "水"]
    prediction = [0 if pred < 0.5 else 1 for pred in logits[:3]]
    for i in range(3):
        if prediction[i] == 0:
            components[i] = ""
            logits[i + 3] = 0
    print("================================================")
    print("预测结果为：{} \n".format(prediction))
    print("混合光谱包含的成分为：{} {} {} \n".format(components[0], components[1], components[2]))
    print("对应浓度分别为：{:.2f} ppm, {:.2f} ppm, {:.2f} ppm".format(logits[3] * 50, logits[4] * 50, logits[5] * 2000))
    print("================================================")

    components = ["甲烷", "丙酮", "水"]
    prediction = [0 if pred < 0.5 else 1 for pred in labs[:3]]
    for i in range(3):
        if prediction[i] == 0:
            components[i] = ""
            labs[i + 3] = 0
    print("================================================")
    print("实际结果为：{} \n".format(prediction))
    print("混合光谱包含的成分为：{} {} {} \n".format(components[0], components[1], components[2]))
    print("对应浓度分别为：{:.2f} ppm, {:.2f} ppm, {:.2f} ppm".format(labs[3] * 50, labs[4] * 50, labs[5] * 2000))
    print("================================================")


