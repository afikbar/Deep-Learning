import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from PIL.Image import BICUBIC

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

# Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, dropout):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes) 

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def evaluate_hw1():
    #Hyper parameters:
    w, h = 14, 14
    input_size = w*h
    hidden_size = 313
    num_classes = 10
    batch_size = 2**9
    dropout_prob = 0.2
    # Load net:
    net = NeuralNet(input_size, num_classes, hidden_size, dropout_prob)
    net.load_state_dict(torch.load('trained_model.pkl'))
    net = to_gpu(net)
    net.eval()

    transform = transforms.Compose([transforms.Resize((w,h), BICUBIC),   
                                    transforms.ToTensor()])

    test_dataset = dsets.FashionMNIST(root='./data',
                            download=True, 
                            train=False, 
                            transform=transform)
    test_size = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
    # Evaluate:
    test_correct = 0
    for images, labels in test_loader:
        images = to_gpu(images.view(-1, 14*14))
        labels = to_gpu(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum()


    test_accuracy = (test_correct.float() / test_size)
    return (test_accuracy).item()
    