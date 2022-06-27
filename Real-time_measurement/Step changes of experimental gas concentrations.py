import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score as R2
import scipy.stats as st
import sys
sys.path.append(r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验")
from utils import array_to_tensor, Data_set, model, hamming_score, predict
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import seaborn as sns


save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\实验数据\已知浓度实验光谱.npy"
noisy_spectra = np.load(save_path)
save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\实验数据\已知浓度实验波数.npy"
nu = np.load(save_path)

inputs = torch.tensor(noisy_spectra, dtype=torch.float32)

CH4_concentration_variations = np.array([0, 3, 5, 10, 13, 15, 20, 25, 30, 35, 40, 45])
H2O_concentration_variations = np.array([2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000, 0])
CH3COCH3_concentration_variations = np.array([50, 45, 40, 35, 30, 20, 0, 10, 25, 30, 35, 40])

CH4_concentration_variations = np.repeat(CH4_concentration_variations, 5)
H2O_concentration_variations = np.repeat(H2O_concentration_variations, 5)
CH3COCH3_concentration_variations = np.repeat(CH3COCH3_concentration_variations, 5)

gas_model = model()
weights_save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\模型保存\model3.pt"
gas_model.load_state_dict(torch.load(weights_save_path, map_location=torch.device('cpu')))
gas_model.eval()


def predict(inputs, model):
    prediction = model(inputs)
    prediction = prediction.cpu().detach().numpy()
    prediction[:, :3] = np.where(prediction[:, :3] > 0.5, 1, 0)
    for i in range(prediction.shape[0]):
        for j in range(3):
            if prediction[i, j] == 0:
                prediction[i, j + 3] = 0
    prediction[:, 3] = prediction[:, 3] * 50
    prediction[:, 4] = prediction[:, 4] * 50
    prediction[:, 5] = prediction[:, 5] * 2000

    return prediction


prediction = predict(inputs, gas_model)


plt.figure()
plt.plot(CH4_concentration_variations, linewidth= 3, color="k", alpha=0.8)
plt.scatter(np.arange(60), prediction[:, 3], edgecolors="darkgreen", color="lime", s=50)


plt.figure()
plt.scatter(np.arange(60), prediction[:, 4], color="k")
plt.plot(CH3COCH3_concentration_variations, color="red")

plt.figure()
plt.scatter(np.arange(60), prediction[:, 5], color="k")
plt.plot(H2O_concentration_variations, color="red")

plt.show()

plt.figure()
# %matplotlib auto
plt.xlim(-1, 50)
plt.ylim(-1, 50)
plt.plot(np.arange(51), np.arange(51), color="k")
plt.scatter(CH4_concentration_variations, prediction[:, 3], s=20, color="red", edgecolors="black", alpha=0.5)

r2 = R2(CH4_concentration_variations, prediction[:, 3])
plt.show()

plt.figure()
# %matplotlib auto
plt.xlim(-1, 50)
plt.ylim(-1, 50)
plt.plot(np.arange(51), np.arange(51), color="k")
plt.scatter(CH3COCH3_concentration_variations, prediction[:, 4], s=20, color="red", edgecolors="black", alpha=0.5)

plt.show()

r2 = R2(CH3COCH3_concentration_variations, prediction[:, 4])
print(r2)

plt.figure()
# %matplotlib auto
plt.xlim(-1, 2000)
plt.ylim(-1, 2000)
plt.plot(np.arange(2001), np.arange(2001), color="k")
plt.scatter(H2O_concentration_variations, prediction[:, 5], s=20, color="red", edgecolors="black", alpha=0.5)

plt.show()

r2 = R2(H2O_concentration_variations, prediction[:, 5])
print(r2)