# DeepLearning
Small deep learning projects for learning, practicing PyTorch and creating my own ChatGPT.

## Table of Contents

- [Image Recognition](#image-recognition)
  - [Multi-layer Perceptron](#multi-layer-perceptron)
  - [Convolutational Neural Nets](#convolutional-neural-networks)
- [Natural Languague Processing](#natural-language-processing)
- [Reinforcement Learning](#reinforcement-learning)


# Image recognition
<sup>[(Back to top)](#table-of-contents)</sup>

Comparison between MLP and CNN

## Multi-Layer Perceptron

This is a sketch of the network architecture used and the data set of handwritten digits.

<p float="middle">
<img src="img/mnist.png" width="350" height="250"/>
<img src="img/hidden_layer.png" width="250" height="250"/>
</p>

This table compares the accuracy obtained (in the validation set) with different learning rates and epochs used in the training process.

| Learning Rate / #Epochs | 1     | 4     | 8     | 16    |
|-------------------------|-------|-------|-------|-------|
| 0.05                    | 91.30% | -- | -- | -- |
| 0.10                    | 93.00% | -- | -- | -- |
| 0.20                    | 94.25% | -- | -- | -- |
| 0.30                    | 95.20% | 97.57% | **98.13%** | 98.11% |
| 0.40                    | 94.78% | -- | -- | -- |

It indicates that the performance begins to stall for learning rates beyond 0.3 and that adding more than 8 epochs does not seem to improve the performance.



## Convolutional Neural Networks

Add some pictures, details and table with results

# Natural Language Processing
<sup>[(Back to top)](#table-of-contents)</sup>

Natural Language Processing

# Reinforcement Learning
<sup>[(Back to top)](#table-of-contents)</sup>

Reinforcement Learning