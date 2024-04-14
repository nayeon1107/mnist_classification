# (2024-1) 인공신경망과 딥러닝 1차 과제 - MNIST Classification

### 📚 File Tree
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

### 📃 Data
 **MNIST Dataset**
![example](https://github.com/nayeon1107/mnist_classification/assets/88521667/b36c37a3-e798-44f1-9d94-351faeb73248)

```bash
# Extract labels by splitting filename
{ID}_{Label}.png  # filename format
```

---

### 🔗 Models 
- Custom MLP
- LeNet5
  - with L2 norm
  - with Dropout


#### Parameters

---

### 📊 Compare Result
#### 🔍 Compare Custom MLP & LeNet5
![LeNet and MLP base](https://github.com/nayeon1107/mnist_classification/assets/88521667/c33dac8c-4e48-4bf4-a327-5da0b2fae9f9)
&nbsp;
#### 🔍 Compare LeNet5(base) & LeNet5(L2 norm) : weight decay = 0.001
![LeNet with L2](https://github.com/nayeon1107/mnist_classification/assets/88521667/aaecb9da-4cbe-4df9-8225-c8b075f00595)
&nbsp;
#### 🔍 Compare LeNet5(base) & LeNet5(Dropout) : dropout = 0.5
![LeNet with Dropout](https://github.com/nayeon1107/mnist_classification/assets/88521667/450f3a17-3f20-4ff4-9837-8e0a59b63c3a)

---

패키지 다운로드
```python
pip install requirements.txt
```

모델 실행
```python
python main.py {model_to_use} {filename_to_save_log}
# ex) python main.py MLP MLP_Base -> Run with model MLP and save log in './MLP_Base.pickle'
# ex) python main.py LeNet5 LeNet_L2 -> Run with model ReNet5 and save log in './LeNet_L2.pickle'
```
