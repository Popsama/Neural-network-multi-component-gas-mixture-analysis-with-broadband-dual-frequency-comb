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
sys.path.append(r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\模拟测试集性能评估")
from local_utils import Convolutional_block, Fully_connected_layer, CNN
from utils import array_to_tensor, Data_set, model, hamming_score, predict
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import seaborn as sns


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


conv = Convolutional_block(16, 7, 7, 7, 8, 64, 64, 128, 0.2)
fc = Fully_connected_layer(64, 32, 0.2)
cnn = CNN(conv, fc)

path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\模型保存\1dcnn.pt"
cnn.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
cnn.eval()

path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\测试集\test_data.npy"
path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\测试集\test_label.npy"
test_input = np.load(path1)
test_label = np.load(path2)


inputs = torch.tensor(test_input, dtype=torch.float32)
print(inputs.shape)
labels = torch.tensor(test_label, dtype=torch.float32)
print(labels.shape)
print(labels.dtype)

pred = cnn(inputs)
pred = pred.cpu().detach().numpy()

with torch.no_grad():
    logits = cnn(inputs)  # logits = torch.Size([1015, 6])
    logits[:, :3] = torch.where(logits[:, :3]>0.5, 1, 0)
    components = ["methan", "acetone", "water vapor"]
    predictions = logits[:, :3].detach().cpu().numpy()
    labels = test_label[:, :3]
    print(predictions.shape)
    print(labels.shape)
    new_pred = transform_labels(predictions)
    new_label = transform_labels(labels)
    cm = confusion_matrix(new_label, new_pred)
    new_pred = transform_labels(predictions)

    xtick = ["methane acetone water", "methane acetone", "methane water", "acetone water",
             "methane", "acetone", "water"]
    ytick = ["methane acetone water", "methane acetone", "methane water", "acetone water",
             "methane", "acetone", "water"]

    plt.figure()
    sns.heatmap(cm, fmt='g', annot=True, cbar=True, xticklabels=xtick, yticklabels=ytick)
    plt.show()


    def relative_error(true, predict):
        error = (np.abs(true - predict) / true)
        return error


    logits = cnn(inputs)  # torch.Size([1015, 6])
    logits = logits.cpu().detach().numpy()
    logits[:, :3] = np.where(logits[:, :3] > 0.5, 1, 0)

    concentration_prediction = np.copy(logits[:, 3:])
    concentration_prediction[:, :2] = concentration_prediction[:, :2] * 50
    concentration_prediction[:, 2] = concentration_prediction[:, 2] * 2000

    concentration_label = np.copy(test_label[:, 3:])
    concentration_label[:, :2] = concentration_label[:, :2] * 50
    concentration_label[:, 2] = concentration_label[:, 2] * 2000

    print("methane r2 is {:.4f} ".format(R2(concentration_label[:, 0], concentration_prediction[:, 0])))
    print("acetone r2 is {:.4f} ".format(R2(concentration_label[:, 1], concentration_prediction[:, 1])))
    print("Water r2 is {:.4f} ".format(R2(concentration_label[:, 2], concentration_prediction[:, 2])))

    index1 = np.argsort(concentration_label[:, 0])
    methane_absolute_error = np.abs(concentration_label[:, 0][index1] - concentration_prediction[:, 0][index1])

    index2 = np.argsort(concentration_label[:, 1])
    actone_absolute_error = np.abs(concentration_label[:, 1][index2] - concentration_prediction[:, 1][index2])

    index3 = np.argsort(concentration_label[:, 2])
    water_absolute_error = np.abs(concentration_label[:, 2][index3] - concentration_prediction[:, 2][index3])

    methane_predicted_concentration = concentration_prediction[:, 0][index1]
    methane_label_concentration = concentration_label[:, 0][index1]
    actone_predicted_concentration = concentration_prediction[:, 1][index2]
    actone_label_concentration = concentration_label[:, 1][index2]
    water_predicted_concentration = concentration_prediction[:, 2][index3]
    water_label_concentration = concentration_label[:, 2][index3]

    print("methane mean absolute error is {:.4f} ppm".format(
        MAE(concentration_label[:, 0], concentration_prediction[:, 0])))
    mean_methane_absolute_error = MAE(concentration_label[:, 0], concentration_prediction[:, 0])
    print("acetone mean absolute error is {:.4f} ppm".format(
        MAE(concentration_label[:, 1], concentration_prediction[:, 1])))
    mean_acetone_absolute_error = MAE(concentration_label[:, 1], concentration_prediction[:, 1])
    print("Water mean absolute error is {:.4f} ppm".format(
        MAE(concentration_label[:, 2], concentration_prediction[:, 2])))
    mean_water_absolute_error = MAE(concentration_label[:, 2], concentration_prediction[:, 2])

    index = np.argsort(concentration_label[:, 0])
    zeros_index = np.argwhere(concentration_label[:, 0][index] == 0)
    cache = concentration_label[:, 0][index]
    cache = cache[zeros_index.shape[0]:]
    cache2 = concentration_prediction[:, 0][index]
    cache2 = cache2[zeros_index.shape[0]:]
    methane_relative_error = relative_error(cache, cache2)
    mean_methane_relative_error = np.mean(methane_relative_error)

    index = np.argsort(concentration_label[:, 1])
    zeros_index = np.argwhere(concentration_label[:, 1][index] == 0)
    cache = concentration_label[:, 1][index]
    cache = cache[zeros_index.shape[0]:]
    cache2 = concentration_prediction[:, 1][index]
    cache2 = cache2[zeros_index.shape[0]:]
    actone_relative_error = relative_error(cache, cache2)
    mean_acetone_relative_error = np.mean(actone_relative_error)
    # index = np.argsort(concentration_label[:, 1])
    # actone_absolute_error = np.abs(concentration_label[:, 1][index] - concentration_prediction[:, 1][index])

    index = np.argsort(concentration_label[:, 2])
    zeros_index = np.argwhere(concentration_label[:, 2][index] == 0)
    cache = concentration_label[:, 2][index]
    cache = cache[zeros_index.shape[0]:]
    cache2 = concentration_prediction[:, 2][index]
    cache2 = cache2[zeros_index.shape[0]:]
    water_relative_error = relative_error(cache, cache2)
    mean_water_relative_error = np.mean(water_relative_error)
    # index = np.argsort(concentration_label[:, 2])
    # water_absolute_error = np.abs(concentration_label[:, 2][index] - concentration_prediction[:, 2][index])

    print(mean_methane_relative_error)
    print(mean_acetone_relative_error)
    print(mean_water_relative_error)

    plt.style.use("ggplot")
    # %matplotlib auto
    # %matplotlib qt5
    plt.figure()

    plt.subplot(3, 3, 1)
    plt.title("methane")
    plt.xlim(0, 50)
    # plt.ylim(0, 50)
    plt.text(10, 40, "R2 = {:.4f}".format(R2(concentration_label[:, 0], concentration_prediction[:, 0])))
    plt.plot(methane_label_concentration, methane_label_concentration, color="k")
    plt.scatter(methane_label_concentration, methane_predicted_concentration, s=20, color="red", edgecolors="black",
                alpha=0.5)

    plt.subplot(3, 3, 2)
    plt.title("methane")
    # plt.yscale("log")
    plt.text(100, 1, "MAE = {:.4f} ppm".format(mean_methane_absolute_error))
    plt.plot(methane_absolute_error[314:], linestyle="--", marker="*")

    plt.subplot(3, 3, 3)
    plt.title("methane")
    # plt.yscale("log")
    plt.text(250, 20, "MRE = {:.4f}%".format(mean_methane_relative_error * 100))
    plt.plot(methane_relative_error[314:] * 100, linestyle="--", marker="*")

    plt.subplot(3, 3, 4)
    plt.title("actone")
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.text(10, 40, "R2 = {:.4f}".format(R2(concentration_label[:, 1], concentration_prediction[:, 1])))
    plt.plot(actone_label_concentration, actone_label_concentration, color="k")
    plt.scatter(actone_label_concentration, actone_predicted_concentration, s=20, color="red", edgecolors="black",
                alpha=0.5)
    print(actone_label_concentration.shape)

    plt.subplot(3, 3, 5)
    plt.title("actone")
    # plt.yscale("log")
    plt.text(100, 1, "MAE = {:.4f} ppm".format(mean_acetone_absolute_error))
    plt.plot(actone_absolute_error[314:], linestyle="--", marker="*")

    plt.subplot(3, 3, 6)
    plt.title("methane")
    # plt.yscale("log")
    plt.text(250, 20, "MRE = {:.4f}%".format(mean_acetone_relative_error * 100))
    plt.plot(actone_relative_error[314:] * 100, linestyle="--", marker="*")

    plt.subplot(3, 3, 7)
    plt.title("water")
    plt.xlim(1000, 2000)
    plt.ylim(500, 2000)
    plt.text(1200, 1800, "R2 = {:.4f}".format(R2(concentration_label[:, 2], concentration_prediction[:, 2])))
    plt.plot(water_label_concentration, water_label_concentration, color="k")
    plt.scatter(water_label_concentration, water_predicted_concentration, s=20, color="red", edgecolors="black",
                alpha=0.5)

    plt.subplot(3, 3, 8)
    plt.title("actone")
    # plt.yscale("log")
    plt.text(100, 50, "MAE = {:.4f} ppm".format(mean_water_absolute_error))
    plt.plot(water_absolute_error[314:], linestyle="--", marker="*")

    plt.subplot(3, 3, 9)
    plt.title("methane")
    # plt.yscale("log")
    plt.text(250, 3, "MRE = {:.4f}%".format(mean_water_relative_error * 100))
    plt.plot(water_relative_error[314:] * 100, linestyle="--", marker="*")

    plt.show()
