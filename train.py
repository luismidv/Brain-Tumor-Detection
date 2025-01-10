import torch
from torch import nn
import  torch.nn.functional as F
import numpy as np
from torchsummary import summary
from torch import optim
import copy
import dataprepare as prp
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms.functional import get_image_num_channels
import random
import wandb




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
        #self.fc3 = nn.Linear(50, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        #X = nn.BatchNorm2d(8)(X)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        #X = nn.BatchNorm2d(16)(X)

        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X,2,2)
        #X = nn.BatchNorm2d(32)(X)

        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X,2,2)
        #X = nn.BatchNorm2d(64)(X)

        X = X.view(-1,self.flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)

        #X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        #X = self.fc3(X)
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
        
        loss.backward()
        optimizer.step()
        accuracy = calculate_accuracy(y_pred, y)
        
        
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
        print(f"Prediction for this run {pred}")
        optimizer.step()
        acc = calculate_accuracy(pred, y)
        loss = loss_fn(pred, y)
        

        epoch_acc += acc.item()
        epoch_loss += loss.item()

    return epoch_acc/len(dataloader), epoch_loss/len(dataloader)

def model_testing(model, input, loss_fn):

    test_transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image_list = ["./selftest/Cancer (8).jpg","./selftest/Cancer (18).jpg", "./selftest/Not Cancer  (3).jpg", "./selftest/Not Cancer  (11).jpg"]
    for image in image_list:
        image = Image.open(image)
        image_resized = test_transformations(image)


        image_resized = image_resized.view(1,3,256,256)
        output = model(image_resized)
        print(f"Output {output}")


    #image_torch = image_torch.view(1,3,256,256)
    #output = model(image_torch)
    #image_torch = test_transformations(image_array)
    #print(f"Checking image type before model {type(image_torch)}")

    #for x,y in input:
        #print(f"Input type {type(x)}\n Input shape {x.shape}")
        #output = model(x)
    #accuracy = calculate_accuracy(output, y)


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
lr = 3e-4
epochs = 60
model = myTumorDetection(params_model)
optimizer = optim.Adam(model.parameters(), lr = lr)
loss_fn = nn.CrossEntropyLoss()
device = torch.device("cpu")

best_valid_loss = float('inf')
lambda1 = lambda epochs: epochs/30
#lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,lambda1)

print(optimizer.state_dict())
for i in range(epochs):
     
     epoch_acc, epoch_loss = model_training(model, train_loader, optimizer, loss_fn,device)
     val_acc, val_loss = model_validation(model, dev_loader, optimizer, loss_fn,device)
     #lr_scheduler.step()
     print(f"Learning rate value {optimizer.state_dict()['param_groups'][0]['lr']}")

     if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), 'tut2-model.pt')
     print(f"____________________________________________________________")
     print(f"Epoch {i}\nTrain acc {epoch_acc * 100} | Train loss {epoch_loss * 100}")
     print(f"Epoch {i}\nVal acc {val_acc * 100} | Val loss {val_loss* 100}")


model.load_state_dict(torch.load('tut2-model.pt'))
model_testing(model,test_loader, loss_fn)


#test_acc, test_loss = model_validation(model, test_loader, optimizer, loss_fn, device)
#print(f"Model final testing\nTest acc {test_acc * 100} | Test loss {test_loss* 100}")
#print(f"____________________________________________________________")


        

        

