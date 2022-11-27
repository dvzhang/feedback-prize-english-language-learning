
# 1. Setup you env
```shell
conda create -n kaggle python=3.7
conda activate kaggle
pip install kaggle

mkdir -p input/commonlitreadabilityprize
```

# 2. Download Data
## 2.1 Download this competition's data from Kaggle API
### 2.1.1 How to use Kaggle api
Firstly, log in to [Kaggle](https://www.kaggle.com/), go to ACCOUT
![pic](./pic/20180502212652888)
Secondly, you can create New Api, getting a kaggle.json file
![pic](./pic/20180502212721622)

Thirdly, copy this file to your home/.kaggle
For example, I copy it to my ~/.kaggle, since I use Ubuntu

### 2.1.2 How to download
Download is easy

```shell
cd input/commonlitreadabilityprize
kaggle competitions download -c commonlitreadabilityprize
unzip commonlitreadabilityprize.zip
```