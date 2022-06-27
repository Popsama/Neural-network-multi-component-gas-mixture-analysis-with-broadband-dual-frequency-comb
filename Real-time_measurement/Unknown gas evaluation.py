from hapi import *
from utils import array_to_tensor, Data_set, model, hamming_score, predict
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def calculate_absorption_spectrum(concentration, path_length,
                                  gas_table,
                                  temperature=296, pressure=1):
    # temperature = K
    # pressure = atm
    volume_portion = concentration / 1000000
    nu, coef = absorptionCoefficient_Voigt(SourceTables=gas_table, HITRAN_units=False,
                                           Environment={"p": pressure, "T": temperature},
                                           WavenumberStep=0.0602,
                                           Diluent={'air': (1 - volume_portion),
                                                    'self': volume_portion})
    coef *= volume_portion
    absorbance = coef * path_length

    return nu, coef, absorbance


db_begin(r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\多组分混合CH3COCH3,CH4,H2O\3组分混合光谱")
CH4_table = 'CH4_2950~3150'
H2O_table = 'H2O_2950~3150'

save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\实验数据\未知浓度实验光谱.npy"
spectra = np.load(save_path)
save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\实验数据\未知浓度实验波数.npy"
wavelength = np.load(save_path)


save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\实验数据\未知浓度实验光谱.npy"
spectra = np.load(save_path)
save_path = r"D:\PYHTON\python3.7\DeepLearningProgram\科研项目\多组分气体识别与浓度检测\数据集\HITRAN_dataset\实验\实验数据\未知浓度实验波数.npy"
wavelength = np.load(save_path)


CH3COCH3 = np.loadtxt(r"D:/PYHTON/python3.7/DeepLearningProgram/科研项目/多组分气体识别与浓度检测/数据集/HITRAN_dataset/丙酮/Acetone/ACETONE_25T.TXT")
nu2 = CH3COCH3[:, 0][55587: 58908][::-1]
coef2 = CH3COCH3[:, 1][55587: 58908][::-1]  # (3321,)

# concentrations predicted by SAM
CH3COCH3_concentration = 43.54
CH4_concentration = 1.3
H2O_concentration = 750


path_length = 580*100
wavenumber_start = 2950
wavenumber_end = 3150
nu1, coef1, CH4_absorp = calculate_absorption_spectrum(CH4_concentration, path_length, CH4_table)
nu3, coef3, H2O_absorp = calculate_absorption_spectrum(H2O_concentration, path_length, H2O_table)

blended_spectra = CH4_absorp[:3321]+ H2O_absorp + (coef2*CH3COCH3_concentration*580)
plt.figure()
plt.xlim(wavenumber_start, wavenumber_end)
plt.plot(nu2, blended_spectra, color="red", label="synthesis")
plt.plot(wavelength, spectra, color="k", label="exp")
plt.legend()
plt.show()