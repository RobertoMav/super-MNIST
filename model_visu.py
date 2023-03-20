import numpy as np 
import math
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch_mnist import NeuralNetwork, test_data, loss_fn

test_data = test_data


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
#model = NeuralNetwork().to(device=device)
model = torch.load("./models/model_adam_softmax.pth")
model.eval()

#Setting loss fnct
def loss_fn(logits, y_true):

    pred_of_truth = logits[0][y_true]
    loss = -math.log(pred_of_truth)

    return loss


## Passing an image through model, just to visualize its acc
def predicting_image(X):
    #Running predict on model
    X = X.to(device)
    logits = model(X)

    #Getting highest prob output
    pred = logits.argmax(1).type(torch.float).item()

    return logits, pred
    
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)   
    

## Set file paths based on added MNIST Datasets

input_path = './data/MNIST/raw'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte')


# Helper function to show a list of images with their relating titles

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(15,15))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 7);     
        plt.axis('off')   
        index += 1
    plt.show()


# Load MINST dataset

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()



images_2_show = []
titles_2_show = []  

wrong_preds = []
images = []
labels = []
predicted = []
lossy = []

for i in range(0, 10000):
    image, label = test_data[i]

    logits, pred = predicting_image(image)

    if int(pred) != y_test[i]:

        loss = loss_fn(logits, y_test[i])

        wrong_preds.append([x_test[i], int(y_test[i]), int(pred), loss])
        images.append(x_test[i])
        labels.append(y_test[i])
        predicted.append(pred)
        lossy.append(loss)
        

error_num = len(wrong_preds)
acc = 100 - error_num/10000*100
print(f"Accuracy (%): {acc:2f}")
print(f"# of wrong predicts: {error_num}")
idx_loss = 25

np_lossy = np.array(lossy)
top_10_loss = np.argsort(np_lossy)[-idx_loss:]

for i in range(0, idx_loss):

    idx = top_10_loss[i]
    images_2_show.append(images[idx])
    titles_2_show.append(f'test = {labels[idx]} pred: {predicted[idx]} loss: {lossy[idx]:2f}')  

show_images(images_2_show, titles_2_show)

