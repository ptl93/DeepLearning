# DeepLearning
This repository contains the implementation of deep learning models for image recognition.

## Environment
- Keras2 + TensorFlow (backend) + Scikit-Learn + Numpy + Pandas + h5py + matplotlib ...

## Structure

The repository-folder are structured by 'problem-set' name, every problem-set contains `data, model` and `source`.

```
├── README.md
├── MNIST
├── Fashion-MNIST
│   ├── data
│   ├── model
│   └── source
├── Simpsons
│   ├── data
│   ├── model
│   └── source
├── StreetViewHouseNumbers
│   ├── data
│   ├── model
│   └── source
├── CIFAR-10
│   ├── data
│   ├── model
│   └── source
```

### Benchmarks

|         Dataset         | Test Accuracy |             Test/Train Info              			|
| :---------------------: | :-----------: | :--------------------------------------: 			|
|          MNIST          |      97,18%   |      10.000/60.0000 samples ratio, 20 epochs        |
|      Fashion-MNIST      |      96,04%   |  	 10.000/50.0000 samples ratio, 50 epochs 		|
| 		 Simpsons		  |      93,31%   |      2527/14.317	samples ratio, 50 epochs		|
|  StreetViewHouseNumbers |      86,38%   | 	 14.894/84.395	samples ratio, 50 epochs	 	|
|  		  CIFAR-10 		  |      83/78%   | 	 10.000/50.000	samples ratio, 60 epochs	 	|



### Running

1. Download & Copy data from download-link.txt in data-fold in  in `<problem-set>/data/`;
2. In Shell change directory into `source` folder, python run `main.py`

```python
cd <problem-set>/source
python main.py
```
  

MNIST until StreetViewHouseNumbers ran on local windows machine:  
Intel(R) Core(TM) i5-7300U CPU @ 2.60GHz, 2.71GHz, 8GB working memory [which took quite a long time, e.g StreetViewHouseNumbers about 24h, Simpsons about 15h,...]
CIFAR-10 ran on [Amazon EC2 Deep Learning Ubuntu image, P3.2xlarge with GPU support] (https://aws.amazon.com/marketplace/pp/B077GCH38C).  
For CIFAR-10 simple_CNN architectute approx. 1h training time. For VGG_modified architectute approx. 2h training time.