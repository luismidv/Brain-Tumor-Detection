import torch
from torch import nn
import  torch.nn.functional as F
import numpy as np
from torchsummary import summary
from torch import optim
import copy
import dataprepare as prp
from tqdm import tqdm




class myTumorDetection(nn.Module):
    def __init__(self, params):
        super(myTumorDetection,self).__init__()

        num_channels, height, width = params['shape_in']
        initial_filters = params['initial_filters']
        num_fc1 = params['num_fc1']
        num_classes = params['num_classes']
        self.dropout_rate = params['dropout_rate']

        
        self.conv1 = nn.Conv2d(num_channels, initial_filters, kernel_size=3)
        height ,width = findConv2dparams(height, width, self.conv1)
        self.conv2 = nn.Conv2d(initial_filters, 2*initial_filters, kernel_size=3)
        height, width = findConv2dparams(height, width, self.conv2)
        self.conv3 = nn.Conv2d(2*initial_filters, 4*initial_filters, kernel_size=3)
        height, width = findConv2dparams(height, width, self.conv3)
        self.conv4 = nn.Conv2d(4*initial_filters, 8*initial_filters, kernel_size=3)
        height, width = findConv2dparams(height, width, self.conv4)
        
        self.flatten = height*width*8*initial_filters

        print(self.flatten)
        self.ly1 = nn.Dropout()
        self.fc1 = nn.Linear(self.flatten, num_fc1)
        self.ly2 = nn.Dropout()
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,self.flatten)
        X = F.dropout(X, self.dropout_rate)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim = 1)
    




def findConv2dparams(height, width, conv1, pool=2):
    
    kernel_size = conv1.kernel_size
    padding = conv1.padding
    stride = conv1.stride
    dilation = conv1.dilation

    hout = np.floor((height+ 2*padding[0]- dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout = np.floor((width+ 2*padding[1]- dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout /= pool
        wout /= pool
        print(hout)
        print(wout)
    return int(hout),int(wout)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1,keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float()/y.shape[0]
    return acc

def model_training(model, dataloader, optimizer, loss_fn, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for (x,y) in tqdm(dataloader, desc = "Training", leave = False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        accuracy = calculate_accuracy(y_pred, y)
        optimizer.step()
        epoch_acc += accuracy.item()
        epoch_loss += loss.item()

    return epoch_acc/len(dataloader), epoch_loss/len(dataloader)

def model_validation(model, dataloader, optimizer,loss_fn,device):
    epoch_acc = 0
    epoch_loss = 0
    model.eval()
    for(x,y) in tqdm(dataloader, desc = "Validation" , leave=False):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)

        optimizer.step()
        acc = calculate_accuracy(pred, y)
        loss = loss_fn(pred, y)

        epoch_acc += acc.item()
        epoch_loss += loss.item()

    return epoch_acc/len(dataloader), epoch_loss/len(dataloader)
        
#prp.load_data('./Brain Tumor Data Set')
train_set, test_set,dev_set = prp.luismi_transformations('./brain')
print(f"Training dataset length {len(train_set)} \n Dev dataset length {len(dev_set)} \n Test dataset length {len(test_set)}")
train_loader, test_loader,dev_loader = prp.data_loaders(train_set, test_set,dev_set)


params_model={
        "shape_in": (3,256,256), 
        "initial_filters": 8,    
        "num_fc1": 100,
        "dropout_rate": 0.25,
        "num_classes": 2
        }

epochs = 5
model = myTumorDetection(params_model)
optimizer = optim.Adam(model.parameters(), lr = 3e-4)
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cpu")

best_valid_loss = float('inf')

for i in range(epochs): 
    epoch_acc, epoch_loss = model_training(model, train_loader, optimizer, loss_fn,device)
    val_acc, val_loss = model_validation(model, dev_loader, optimizer, loss_fn,device)

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
    print(f"Epoch {i}\nTrain acc {epoch_acc} | Train loss {epoch_loss}")
    print(f"Epoch {i}\nVal acc {val_acc} | Val loss {val_loss}")
    print(f"____________________________________________________________")
    model.load_state_dict(torch.load('tut2-model.pt'))
    test_acc, test_loss = model_validation(model, test_loader, optimizer, loss_fn, device)
    print(f"Model final testing\nTest acc {test_acc} | Test loss {test_loss}")



        

        
