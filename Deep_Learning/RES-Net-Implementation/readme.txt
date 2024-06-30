RESNet Implementation on Image Data
===========================================
(Jupyter Notebook to be run Sequentially)

In this project, I have developed an automated framework for training and evaluating RESNet model.

The full functionality of PyTorch is used to develop the complete pipeline starting from data preparation to the final model evaluation.

Dataset: FashionMNIST

Folder Structure:
1) cnn.ipynb: RESNet Model Implementation
2) scripts/model_pipeline.py: Contains all functions used for model training, evaluation, plotting, etc.

NOTE: The code runs on GPU. Those settings are configured in the backend file model_pipeline.py. While running the cells in this notebook, wherever there is device paramether, please change the parameter according to the OS('cuda' or 'cpu') this file is being run on. Right now, it is set to 'mps' since I have Mac OS.

Overview:
=========

CNN with Residual Blocks (RESNet):
-----------------------------------

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