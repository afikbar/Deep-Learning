
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.fc = nn.Linear(512,10)
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(1)
        
        
    def forward(self, x):
        out = x
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.maxpool(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.maxpool(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.maxpool(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def evaluate_hw2():
    #Hyper parameters:
    num_epochs = 2000
    batch_size = 64
    lr = 0.001
    
    # Load net:
    net = CNN()
    net.load_state_dict(torch.load('model.pkl',map_location=lambda storage, loc: storage))
    net = to_gpu(net)
    net.eval()

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4942142800245098, 0.4851313890165441, 0.4504090927542892),
                                         (0.24665251509498004, 0.24289226346005355, 0.26159237802202373)),
    ])

    test_dataset = dsets.CIFAR10(root='./data',
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
        images = to_gpu(images)
        labels = to_gpu(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum()

    test_accuracy = (test_correct.float() / test_size)
    return (test_accuracy).item()

