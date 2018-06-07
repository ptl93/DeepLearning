# DeepLearning
This repository contains the implementation of deep learning models.

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
```

### Benchmarks

|         Dataset         | Test Accuracy |             Test/Train Info              			|
| :---------------------: | :-----------: | :--------------------------------------: 			|
|          MNIST          |      97,18%   |      10.000/60.0000 samples ratio, 20 epochs        |
|      Fashion-MNIST      |      96,04%   |  	 10.000/50.0000 samples ratio, 50 epochs 		|
| 		 Simpsons		  |      93,31%   |      2527/14317		samples ratio, 50 epochs		|
|  StreetViewHouseNumbers |      86,38%   | 	 14894/84395	samples ratio, 50 epochs	 	|



### Running

1. Download & Copy data from download-link.txt in data-fold in  in `<problem-set>/data/`;
2. In Shell change directory into  `source` folder, python run `main.py`

```python
cd <problem-set>/source
python main.py
```
