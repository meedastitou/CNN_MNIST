# Lab 2: Deep Learning with PyTorch



## Tasks

### Part 1: CNN Classifier

#### Dataset
- **MNIST Dataset**: [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

#### Steps
1. **CNN Architecture**:
   - Build a CNN classifier using PyTorch to classify the MNIST dataset.
   - Define layers such as convolutional layers, pooling layers, and fully connected layers.
   - Configure hyperparameters, including kernels, padding, stride, optimizers, and regularization.
   - the model runs in GPU mode.

2. **Faster R-CNN**:
   - Repeat the classification task using Faster R-CNN.
  
3. **Comparison**:
   - CNN           :  Accuracy: 0.9857, F1 Score: 0.9857
   - Faster R-CNN  :  Accuracy: , F1 Score: 

5. **Pretrained Models**:
   - Fine-tune pretrained models (VGG16 and AlexNet) on the MNIST dataset.
   - VGG16         :  Accuracy: 0.9942 F1 Score: 0.9942
   - alexnet       :  Accuracy: 0.9941 F1 Score: 0.9941

---

### Part 2: Vision Transformer (ViT)

#### Steps
1. **ViT Architecture**:
   - Follow this tutorial to establish a ViT model architecture from scratch: [Vision Transformers Tutorial](https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c).
   - Use the ViT model to perform a classification task on the MNIST dataset.

2. **Result Analysis**:
   - training loop is experiencing very little change in the loss values over multiple epochs, which suggests that the model is not learning effectively.
     
     ![image](https://github.com/user-attachments/assets/227cd60d-6a8c-4531-9cec-4afb2837c5a9)
   


---


