
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
    
# Hyper Parameters
num_epochs = 2000
batch_size = 64
lr = 0.001

# Image Preprocessing 
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
                             (0.2470322324632819, 0.24348512800005573, 0.26158784172796434)),
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4942142800245098, 0.4851313890165441, 0.4504090927542892),
                             (0.24665251509498004, 0.24289226346005355, 0.26159237802202373)),
    ])

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='./data', 
                            train=True, 
                            transform=train_transform,
                            download=True)

test_dataset = dsets.CIFAR10(root='./data', 
                           train=False, 
                           transform=test_transform)

class_names = train_dataset.classes

train_size = len(train_dataset)
test_size = len(test_dataset)

# Dataset Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size,
                                           num_workers=8,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size,
                                          num_workers=8,
                                          shuffle=False)

net = to_gpu(CNN())
print(f'Num of trainable parameters : {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

# Loss and Optimizer
# Softmax is internally computed.
criterion = to_gpu(nn.CrossEntropyLoss())
optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

error_train = []
error_test = []
loss_train = []
loss_test = []

# Training the Model
for epoch in range(num_epochs):
    net.train()
    train_correct = 0
    batch_losses = []
    for i, (images, labels) in enumerate(train_loader):
        images = to_gpu(images)
        labels = to_gpu(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        batch_losses.extend([train_loss.item()]*len(images))

        # Accuracy:
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum()
    train_accuracy = (train_correct.float() / train_size)
    epoch_train_loss = sum(batch_losses) / train_size
    loss_train.append(epoch_train_loss)
    error_train.append(1-train_accuracy)

    net.eval()   
    test_correct = 0
    batch_losses = []
    for images, labels in test_loader:
        images = to_gpu(images)
        labels = to_gpu(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum()

#         Test Loss:
        test_loss = criterion(outputs, labels)
        batch_losses.extend([test_loss.item()]*len(images))

    test_accuracy = (test_correct.float() / test_size)
    error_test.append(1 - test_accuracy)
    loss_test.append(sum(batch_losses) / test_size)
    
    print(f"Epoch: [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {100.0*train_accuracy:.2f}, Test Accuracy: {100.0*test_accuracy:.2f}")


# Plot convergence:
import matplotlib.pyplot as plt
import seaborn as sn

sn.set()
ind = list(range(1, num_epochs+1))
# Error
plt.plot(ind, error_train, label='Train')
plt.plot(ind, error_test, label='Test')
plt.title('Error-rate during epochs')
plt.xlabel('Epochs')
plt.ylabel('Error-rate')
plt.legend()
plt.show()

# Loss
plt.plot(ind, loss_train, label='Train')
plt.plot(ind, loss_test, label='Test')
plt.title('Loss during epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

