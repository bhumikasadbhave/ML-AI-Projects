Multi-layer Perceptron and Convolutional Neural Network Implementation and Comparison on Image Data
=========================================================================================================
(Jupyter Notebook to be run Sequentially)

In this project, I have developed an automated framework for training and evaluating basic NN models.

The full functionality of PyTorch is used to develop the complete pipeline starting from data preparation to the final model evaluation.

Dataset: FashionMNIST

Folder Structure:
1) mlp.ipynb: MLP Models
2) cnn.ipynb: CNN Models
3) scripts/model_pipeline.py: Contains all functions used for model training, evaluation, plotting, etc.

NOTE: All the CNN file code runs on GPU. Those settings are configured in the backend file model_pipeline.py. While running the cells in this notebook, wherever there is device paramether, please change the parameter according to the OS('cuda' or 'cpu') this file is being run on. Right now, it is set to 'mps' since I have Mac OS.

Overview:
=========

MLP Models:
--------------
1. Simple MLP with Non linearities:
   - Input: 784 features (e.g., flattened 28x28 image).
   - Layers:
	Linear layer (784 -> 512)
	Linear layer (512 -> 512)
	ReLU activation
	Linear layer (512 -> 100)
	ReLU activation
	Linear layer (100 -> 100)
	Linear layer (100 -> 10)

2. MLP with non-linearities and dropout
  - Input: 784 features (e.g., flattened 28x28 image).
  - Layers:
	Linear layer (784 -> 512)
	ReLU activation
	Dropout layer (p=0.2)
	Linear layer (512 -> 512)
	ReLU activation
	Dropout layer (p=0.2)
	Linear layer (512 -> 10)

3. MLP with non-linearities, dropout and batch norm
   - Input: 784 features (e.g., flattened 28x28 image).
   - Layers:
    	Input layer (784 -> 512)
    	Batch Normalization (512)
    	LeakyReLU activation
    	Linear layer (512 -> 512)
    	Batch Normalization (512)
    	LeakyReLU activation
    	Dropout layer (p=0.2)
    	Linear layer (512 -> 512)
    	Batch Normalization (512)
    	LeakyReLU activation
    	Dropout layer (p=0.2)
    	Linear layer (512 -> 512)
    	Batch Normalization (512)
    	LeakyReLU activation
    	Output layer (512 -> 10)


CNN Models:
--------------
1. CNN Model 1: Simple model with conv2d, max pooling and batch norm layers

Model Architecture:
- Input: 1 channel (e.g., grayscale image)
- Conv2d Block 1
    - Convolutional Layer: (1 -> 32, 3x3 kernel, stride 1, padding 1)
    - Batch Normalization: (32 features)
    - ReLU Activation
    - Max Pooling: (2x2 kernel, stride 2)
- Conv2d Block 2
    - Convolutional Layer: (32 -> 64, 3x3 kernel, stride 1, padding 1)
    - Batch Normalization: (64 features)
    - ReLU Activation
    - Max Pooling: (2x2 kernel, stride 2)
- Conv2d Block 3
    - Convolutional Layer: (64 -> 128, 3x3 kernel, stride 1, padding 1)
    - Batch Normalization: (128 features)
    - ReLU Activation
    - Max Pooling: (2x2 kernel, stride 2)
- Fully Connected Block
    - Linear Layer: (1152 -> 256)
    - ReLU Activation
    - Dropout: (p=0.2)
    - Linear Layer: (256 -> 10)


2. CNN with Residual Blocks:

Model Architecture:

Residual Block: 
- Convolutional Layer 1: (32 -> 64, 3x3 kernel, stride 1, padding 'same')
- Batch Normalization: (64 features)
- ReLU Activation
- Convolutional Layer 2: (64 -> 64, 3x3 kernel, stride 1, padding 'same')
- Batch Normalization: (64 features)
- ReLU Activation
- Dropout: (p=0.2)
- Max Pooling: (2x2 kernel, stride 2)
- Residual Connection: (32 -> 64, 1x1 kernel, stride 2, padding 'valid')

CNN Model:
- Input: 1 channel (e.g., grayscale image)
    - Conv2d Block
    - Convolutional Layer: (1 -> 32, 3x3 kernel, stride 1, padding 'same')
    - Batch Normalization: (32 features)
    - ReLU Activation
- Residual Block 1
- Residual Block 2
- Average Pooling
    - Avg Pooling: (2x2 kernel, stride 2)
- Fully Connected Block
    - Dropout: (p=0.2)
    - Linear Layer: (576 -> 10)
    - ReLU Activation