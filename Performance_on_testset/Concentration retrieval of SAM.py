import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
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


def relative_error(true, predict):
    error = (np.abs(true-predict)/true)
    return error


Gpu = torch.device("cuda")
gas_model = model().to(Gpu)
weights_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\模型保存\model3.pt"
gas_model.load_state_dict(torch.load(weights_save_path))
gas_model.eval()

path1 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\测试集\test_data.npy"
path2 = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\甲烷、丙酮、水数据库\数据集\混合气体吸收光谱\测试集\test_label.npy"
test_input = np.load(path1)
test_label = np.load(path2)

inputs = array_to_tensor(test_input)
labels = array_to_tensor(test_label)

logits = gas_model(inputs)  # torch.Size([1015, 6])
logits = logits.cpu().detach().numpy()
logits[:, :3] = np.where(logits[:, :3]>0.5, 1, 0)


concentration_prediction = np.copy(logits[:, 3:])
concentration_prediction[:, :2] = concentration_prediction[:, :2]*50
concentration_prediction[:, 2] = concentration_prediction[:, 2]*2000

concentration_label = np.copy(test_label[:, 3:])
concentration_label[:, :2] = concentration_label[:, :2]*50
concentration_label[:, 2] = concentration_label[:, 2]*2000

print("methane r2 is {:.4f} ".format(R2(concentration_label[:, 0], concentration_prediction[:, 0])))
print("acetone r2 is {:.4f} ".format(R2(concentration_label[:, 1], concentration_prediction[:, 1])))
print("Water r2 is {:.4f} ".format(R2(concentration_label[:, 2], concentration_prediction[:, 2])))


index = np.argsort(concentration_label[:, 0])
methane_absolute_error = np.abs(concentration_label[:, 0][index] - concentration_prediction[:, 0][index])

index = np.argsort(concentration_label[:, 1])
actone_absolute_error = np.abs(concentration_label[:, 1][index] - concentration_prediction[:, 1][index])

index = np.argsort(concentration_label[:, 2])
water_absolute_error = np.abs(concentration_label[:, 2][index] - concentration_prediction[:, 2][index])

methane_predicted_concentration = concentration_prediction[:, 0][index]
methane_label_concentration = concentration_label[:, 0][index]
actone_predicted_concentration = concentration_prediction[:, 1][index]
actone_label_concentration = concentration_label[:, 1][index]
water_predicted_concentration = concentration_prediction[:, 2][index]
water_label_concentration = concentration_label[:, 2][index]


print("methane mean absolute error is {:.4f} ppm".format(MAE(concentration_label[:, 0], concentration_prediction[:, 0])))
mean_methane_absolute_error = MAE(concentration_label[:, 0], concentration_prediction[:, 0])
print("acetone mean absolute error is {:.4f} ppm".format(MAE(concentration_label[:, 1], concentration_prediction[:, 1])))
mean_acetone_absolute_error = MAE(concentration_label[:, 1], concentration_prediction[:, 1])
print("Water mean absolute error is {:.4f} ppm".format(MAE(concentration_label[:, 2], concentration_prediction[:, 2])))
mean_water_absolute_error = MAE(concentration_label[:, 2], concentration_prediction[:, 2])


index = np.argsort(concentration_label[:, 0])
methane_absolute_error = np.abs(concentration_label[:, 0][index] - concentration_prediction[:, 0][index])

index = np.argsort(concentration_label[:, 1])
actone_absolute_error = np.abs(concentration_label[:, 1][index] - concentration_prediction[:, 1][index])

index = np.argsort(concentration_label[:, 2])
water_absolute_error = np.abs(concentration_label[:, 2][index] - concentration_prediction[:, 2][index])


index = np.argsort(concentration_label[:, 0])
zeros_index = np.argwhere(concentration_label[:, 0][index]==0)
cache = concentration_label[:, 0][index]
cache = cache[zeros_index.shape[0]:]
cache2 = concentration_prediction[:, 0][index]
cache2 = cache2[zeros_index.shape[0]:]
methane_relative_error = relative_error(cache, cache2)
mean_methane_relative_error = np.mean(methane_relative_error)


index = np.argsort(concentration_label[:, 1])
zeros_index = np.argwhere(concentration_label[:, 1][index]==0)
cache = concentration_label[:, 1][index]
cache = cache[zeros_index.shape[0]:]
cache2 = concentration_prediction[:, 1][index]
cache2 = cache2[zeros_index.shape[0]:]
actone_relative_error = relative_error(cache, cache2)
mean_acetone_relative_error = np.mean(actone_relative_error)
# index = np.argsort(concentration_label[:, 1])
# actone_absolute_error = np.abs(concentration_label[:, 1][index] - concentration_prediction[:, 1][index])

index = np.argsort(concentration_label[:, 2])
zeros_index = np.argwhere(concentration_label[:, 2][index]==0)
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
plt.ylim(0, 50)
plt.text(10, 40, "R2 = {:.4f}".format(R2(concentration_label[:, 0], concentration_prediction[:, 0])))
plt.plot(methane_label_concentration, methane_label_concentration, color="k")
# plt.scatter(methane_label_concentration, methane_predicted_concentration, s=20, color="red", edgecolors="black", alpha=0.5)
plt.scatter(methane_label_concentration[257:], methane_predicted_concentration[257:], s=20, color="red", edgecolors="black", alpha=0.5)

plt.subplot(3, 3, 2)
plt.title("methane")
# plt.yscale("log")
plt.text(100, 1, "MAE = {:.4f} ppm".format(mean_methane_absolute_error))
plt.plot(methane_absolute_error, linestyle="--", marker="*")

plt.subplot(3, 3, 3)
plt.title("methane")
# plt.yscale("log")
plt.text(250, 20, "MRE = {:.4f}%".format(mean_methane_relative_error*100))
plt.plot(methane_relative_error*100, linestyle="--", marker="*")

plt.subplot(3, 3, 4)
plt.title("actone")
plt.xlim(0, 50)
plt.ylim(0, 50)
plt.text(10, 40, "R2 = {:.4f}".format(R2(concentration_label[:, 1], concentration_prediction[:, 1])))
plt.plot(actone_label_concentration, actone_label_concentration, color="k")
plt.scatter(actone_label_concentration[257:], actone_predicted_concentration[257:], s=20, color="red", edgecolors="black", alpha=0.5)
print(actone_label_concentration[257:].shape)


plt.subplot(3, 3, 5)
plt.title("actone")
# plt.yscale("log")
plt.text(100, 1, "MAE = {:.4f} ppm".format(mean_acetone_absolute_error))
plt.plot(actone_absolute_error, linestyle="--", marker="*")


plt.subplot(3, 3, 6)
plt.title("methane")
# plt.yscale("log")
plt.text(250, 20, "MRE = {:.4f}%".format(mean_acetone_relative_error*100))
plt.plot(actone_relative_error*100, linestyle="--", marker="*")

plt.subplot(3, 3, 7)
plt.title("water")
plt.xlim(1000, 2000)
plt.ylim(1000, 2000)
plt.text(1200, 1800, "R2 = {:.4f}".format(R2(concentration_label[:, 2], concentration_prediction[:, 2])))
plt.plot(water_label_concentration[257:], water_label_concentration[257:], color="k")
plt.scatter(water_label_concentration[257:], water_predicted_concentration[257:], s=20, color="red", edgecolors="black", alpha=0.5)
print(water_label_concentration[257:].shape)

plt.subplot(3, 3, 8)
plt.title("actone")
# plt.yscale("log")
plt.text(100, 50, "MAE = {:.4f} ppm".format(mean_water_absolute_error))
plt.plot(water_absolute_error, linestyle="--", marker="*")

plt.subplot(3, 3, 9)
plt.title("methane")
# plt.yscale("log")
plt.text(250, 3, "MRE = {:.4f}%".format(mean_water_relative_error*100))
plt.plot(water_relative_error*100, linestyle="--", marker="*")

plt.show()