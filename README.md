# EE559-MiniProjet2

## Code structure :

The project is structured in three main files, and a notebook that contains further plots. 

### `Module.py`
The file __Module.py__ contains the definition of the skeleton class Module and the implementation of all the modules that are asked in this project, i.e : 
- Linear
- Tanh
- ReLU
- Sequential
- LossMSE

We also implemented further structures than those that were asked in the project description. Those are the extra modules, that fall into three main categories, namely :

- Loss Modules (L1 and CrossEntropy)
- Activation Modules (Sigmoid, Leaky Relu , Tanshrink and Threshold)
- The Dropout Module

### `utils.py`
This file contains utilitary methods that are used to lighten the code for test.py, but also the code that generated all the plots that can be found in the report.

### `Optimizer.py`
An abstract class defining what an optimizer should look like, if one decides to build beyond our framework. Inspired by the pytorch documentation.


### `test.py`
* Generates nbPoints for the train set and the test set, as required in the problem statement. 
* Normalizes the data.
* Build one model of the form Sequential : Linear(2,25)->ReLU -> Linear(25,25) -> Relu -> Linear(25,25) -> Relu -> Linear(25,2) -> Tanh
* Other activations can be used, and Dropout can be added just before the output layer
* Epochs are set to 100 (can be tuned)
* Learning rate set to 5e-1 (can be tuned)
* Mini batch size set to 50 (can be tuned)
* Criterion is LossMSE (by adding a Boolean to have the sum of squared errors instead of Mean Squared Error), can be changed for L1Loss or CrossEntropyLoss

To run the code, run the command :

```console
python3 test.py
```
This prints the value of the final train error and the test error, and logs the the training errors for each batch in a file error_logs.txt


### `Project_2.ipynb`

The notebook contains all the code of the three above files into one single notebook, with extra visualizations.