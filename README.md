

# Neural network multi-component gas mixture analysis with broadband dual-frequency comb absorption spectroscopy

Code repository for paper named "Neural network multi-component gas mixture analysis with broadband dual-frequency comb absorption spectroscopy"

<!-- PROJECT SHIELDS -->

![gasmixture](https://img.shields.io/badge/Gas%20Mixture-Methane%2C%20Acetone%20%26%20Water-brightgreen.svg)
![Modification](https://img.shields.io/badge/Lastmodified-Today-brightgreen.svg)
![Author](https://img.shields.io/badge/Author-Linbo%20Tian-orange.svg)
![MIT License](https://img.shields.io/apm/l/vim-mode.svg)
![features](https://img.shields.io/badge/Neural%20Network-Overlapping%20absorption%20features-blue.svg)


<p align="center">
  <a href="https://github.com/Popsama/Neural-network-multi-component-gas-mixture-analysis-with-broadband-dual-frequency-comb/blob/main/logo.png">
    <img src="https://github.com/Popsama/Neural-network-multi-component-gas-mixture-analysis-with-broadband-dual-frequency-comb/blob/main/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Correponding source code</h3>
  <p align="center">
    for SAM, 1D-CNN & 2LARNN for multi-component gas mixture analysis
    <br />
    <a href="https://github.com/Popsama/Spectra-analysis-model-of-Gas-mixture-with-overlapping-absorption-features"><strong>Documents »</strong></a>
    <br />
    <br />
    <a href="https://github.com/Popsama/Spectra-analysis-model-of-Gas-mixture-with-overlapping-absorption-features">查看Demo</a>
    ·
    <a href="https://github.com/Popsama/Spectra-analysis-model-of-Gas-mixture-with-overlapping-absorption-features/issues">Propose Bug</a>
    ·
    <a href="https://github.com/Popsama/Spectra-analysis-model-of-Gas-mixture-with-overlapping-absorption-features/issues">Propose new features</a>
  </p>

</p>


 
## Including

- [Datasets](#Datasets)
  - [Experimental datasets](#Experimental datasets)
  - [Simulation datasets](#Simulation datasets)
- [Model_architecture](#Model_architecture)
  - [Model architecture tuning of SAM](#Model architecture tuning of SAM.py)
  - [Training parameters tuning of SAM](#Training parameters tuning of SAM.py)
  - [1D_CNN](#1D_CNN.ipynb)
  - [2LARNN](#2LARNN.ipynb)
- [Performance_on_testset](#Performance_on_testset)
- [Real-time_measurement](#Real-time_measurement)


## Directory

```
filetree 
├── README.md
├── LICENSE.txt
├── Datasets
│  ├── Experimental datasets
│  │  ├── Step change of experimental data
│  │  └── Unknown ambient evaluation
│  ├── Simulation datasets
│  │  ├── Train+val
│  │  └── Test
├── Model architecture
│  ├── 1D_CNN.ipynb
│  ├── 2LARNN.ipynb
│  ├── Model architecture turning of SAM.py
│  ├── Training parameters turning of SAM.py
├── Performance_on_testset
├── Real-time_measurement
└── util.py

```

### 作者
Linbo Tian
tianlinbodawang@163.com
201920409@sdu.edu.cn
 
 *您也可以在贡献者名单中参看所有参与该项目的开发者。*

### 版权说明

该项目签署了MIT 授权许可，详情请参阅 [LICENSE.txt](https://github.com/Popsama/Spectra-analysis-model-of-Gas-mixture-with-overlapping-absorption-features/master/LICENSE.txt)






