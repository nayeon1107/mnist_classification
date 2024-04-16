# MNIST Classification

## 📚 파일 구조
```bash
📂 mnist_classification
 ┣ 📂 templates
 ┃ ┗ 📜 dataset.py
 ┃ ┗ 📜 main.py
 ┃ ┗ 📜 model.py
 ┣ 📂 data
 ┃ ┗ 📂 test
 ┃ ┗ 📂 train
 ┃ ┃ ┗ 📷 00000_5.png
 ┃ ┃ ┗ 📷 00001_0.png
 ┃ ┃ ┗ 📷 00002_4.png
 ┗ ┗ ┗ ...
```

## 📃 Data
 **MNIST Dataset**
![example](https://github.com/nayeon1107/mnist_classification/assets/88521667/b36c37a3-e798-44f1-9d94-351faeb73248)

```bash
# Extract labels by splitting filename
{ID}_{Label}.png  # filename format
```

---

## 🔗 Models 
- Custom MLP (base)
- LeNet5 (base)
  - with Dropout
  - with BatchNormalization


### Parameters
 **1. ReNet5's parameter -> total 44,426**

 
![LeNet parameter](https://github.com/nayeon1107/mnist_classification/assets/88521667/7f6338ca-894e-4974-a027-ae05a734d4b4)

 

 **2. MLP's parameter -> total 45,042**

 
![MLP Parameter](https://github.com/nayeon1107/mnist_classification/assets/88521667/0d72e9c1-242a-4c77-981b-63171257e569)


&nbsp; &nbsp;


## 📊 Compare Result
### 🔍 Compare Custom MLP & LeNet5
![LeNet and MLP base](https://github.com/nayeon1107/mnist_classification/assets/88521667/e42ce18a-d1ff-4e29-8f4c-5bc9e4998c0b)
```bash
▶ LeNet5 모델이 일반적으로 약 7%P 높은 성능을 보임
```
&nbsp; &nbsp;
### 🔍 Compare LeNet5(base) & LeNet5(Dropout) : dropout = 0.5
![LeNet with Dropout](https://github.com/nayeon1107/mnist_classification/assets/88521667/399099f4-53e8-4139-a93c-495dc574a843)
```bash
▶ 각 Fully Connected layer 에 Dropout(0.5)을 적용한 결과 Test set에서 약 0.5%P 전후로 정확도가 상승하였으며, Loss 값도 소폭 감소함
▶ Overfitting 을 방지하고 일반화 성능을 높였다고 할 수 있음
```
&nbsp; &nbsp;
### 🔍 Compare LeNet5(base) & LeNet5(BatchNormalization)
![LeNet with BatchNorm](https://github.com/nayeon1107/mnist_classification/assets/88521667/644b9c3f-35e8-49bc-820b-c5fb3b541432)
```bash
▶ 각 Convolution 층에 BatchNormalization을 적용한 결과 Test set에서 약 0.5%P 전후로 정확도가 상승하였으며, Loss 값도 소폭 감소함
▶ Overfitting 을 방지하고 일반화 성능을 높였다고 할 수 있음
```
---

패키지 다운로드
```python
pip install requirements.txt
```

모델 실행
```python
python main.py {model_to_use} {filename_to_save_log}

# ex) python main.py MLP MLP_Base -> Run with model MLP and save log in './MLP_Base.pickle'
# ex) python main.py LeNet5 LeNet_Dropout -> Run with model ReNet5 and save log in './LeNet_Dropout.pickle'
```
