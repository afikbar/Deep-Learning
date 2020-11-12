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
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_prob):
        super(NeuralNet, self).__init__()
        self.hiddens = [nn.Linear(input_size, hidden_sizes[0])] + [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(0, len(hidden_sizes)-1)]
        self.hiddens = nn.ModuleList(self.hiddens)
        self.last = nn.Linear(hidden_sizes[-1], num_classes) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        out = x
        for layer in self.hiddens:
            out = layer(out)
            out = self.dropout(out)
            out = self.relu(out)

        out = self.last(out)
        return out

# Hyper Parameters
w, h = 14, 14
input_size = w*h
hidden_sizes = [313]

num_classes = 10
num_epochs = 1000
batch_size = 2**9
learning_rate = 0.001
# momentum = 0
# nesterov = False
dropout_prob = 0.2
weight_decay =  0.01


transform = transforms.Compose([transforms.Resize((w,h), BICUBIC),   
                                transforms.ToTensor(),])

# FashionMNIST Dataset (Images and Labels)
train_dataset = dsets.FashionMNIST(root='./data', 
                            train=True, 
                            transform=transform,
                            download=True)

test_dataset = dsets.FashionMNIST(root='./data', 
                           train=False, 
                           transform=transform)

class_names = train_dataset.classes

train_size = len(train_dataset)
test_size = len(test_dataset)

# Dataset Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


net = NeuralNet(input_size, hidden_sizes, num_classes, dropout_prob)
net = to_gpu(net)
# Loss and Optimizer
# Softmax is internally computed.
criterion = to_gpu(nn.CrossEntropyLoss())
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, nesterov=nesterov)
optimizer = torch.optim.AdamW(net.parameters(),lr=learning_rate, weight_decay=weight_decay)

print(f'Num of trainable parameters : {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

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
        images = to_gpu(images.view(-1, 14*14))
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
        images = to_gpu(images.view(-1, 14*14))
        labels = to_gpu(labels)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum()

        #Test Loss:
        test_loss = criterion(outputs, labels)
        batch_losses.extend([test_loss.item()]*len(images))

    test_accuracy = (test_correct.float() / test_size)
    error_test.append(1 - test_accuracy)
    loss_test.append(sum(batch_losses) / test_size)
    
    print(f"Epoch: [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Test Accuracy: {100.0*test_accuracy:.4f}")


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