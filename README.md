# MNIST Handwritten Digit Classification

This Python algorithm will be classifying handwritten digits ranging from 0-10. There are 2 files in this repository and both files will be using PyTorch framework. One file will be using traditional Neural Network and the other file will be using Convolutional Neural Network. **Convolutional Neural Network (CNN)** often times produce a model that yields more accuracy and can be trained much more quicker than traditional Neural Network.

There are 4 images in this directory, 2 **Loss Against Epoch** graphs and 2 **Accruacy Against Epoch** graphs. You can see that the algorithm with CNN yield a model with **4%** more accuracy than the other. You can play around with the hyperparameters such as **Number of channels, Kernel Size, Stride & Padding** to increase the accuracy and decrease the loss of your model.

If you have a **NVIDIA GPU** that supports **CUDA Cores**, feel free to uncomment the codes to run it on your GPU. Training the model using your GPU makes it consume less time compared to training the model using your CPU.