#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import time
import random
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from advertorch.attacks import LinfPGDAttack
import sklearn.metrics as metrics


# In[3]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[4]:


config = dict(
    saved_path="saved_models/random.pt",
    best_saved_path = "saved/random_best.pt",
    lr=0.001, 
    EPOCHS = 3,
    BATCH_SIZE = 32,
    IMAGE_SIZE = 224,
    TRAIN_VALID_SPLIT = 0.2,
    device=device,
    SEED = 42,
    pin_memory=True,
    num_workers=2,
    USE_AMP = True,
    channels_last=False)


# In[5]:


random.seed(config['SEED'])
# If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG 
np.random.seed(config['SEED'])
# Prevent RNG for CPU and GPU using torch
torch.manual_seed(config['SEED'])
torch.cuda.manual_seed(config['SEED'])
torch.backends.cudnn.benchmarks = True
torch.backends.cudnn.deterministic = True

torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


# In[6]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'],config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[7]:


train_data = torchvision.datasets.CIFAR10(root='../Images', train=True,
                                        download=True, transform=data_transforms['train'])
train_dl = torch.utils.data.DataLoader(train_data, batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])

test_data = torchvision.datasets.CIFAR10(root='../Images', train=False,
                                       download=True, transform=data_transforms['test'])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True, num_workers = config['num_workers'],
                                          pin_memory = config['pin_memory'])
valid_dl = test_dl


# In[8]:


valid_data = test_data


# In[9]:


import matplotlib.pyplot as plt
a = iter(valid_dl)
b = next(a)
print(b[1])
plt.imshow(b[0][0][0])



mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.classifier[3] = nn.Linear(in_features = 1024, out_features = 10, bias = True)

# Initialize the model and set it to the GPU if available
clf = mobilenet


# # Attack

# In[12]:


def train_model(model,criterion,optimizer,train_dl,valid_dl,num_epochs=10):

    since = time.time()                                            
    batch_ct = 0
    example_ct = 0
    best_acc = 0.3
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        run_corrects = 0
        #Training
        model.train()
        for x,y in train_dl: #BS=32 ([BS,3,224,224], [BS,4])            
            if config['channels_last']:
                x = x.to(config['device'], memory_format=torch.channels_last) #CHW --> #HWC
            else:
                x = x.to(config['device'])
            y = y.to(config['device']) #CHW --> #HWC
            
            
            
            optimizer.zero_grad()
            #optimizer.zero_grad(set_to_none=True)
            ######################################################################
            
            train_logits = model(x) #Input = [BS,3,224,224] (Image) -- Model --> [BS,4] (Output Scores)
            
            _, train_preds = torch.max(train_logits, 1)
            train_loss = criterion(train_logits,y)
            train_loss = criterion(train_logits,y)
            run_corrects += torch.sum(train_preds == y.data)
            
            train_loss.backward() # Backpropagation this is where your W_gradient
            loss=train_loss

            optimizer.step() # W_new = W_old - LR * W_gradient 
            example_ct += len(x) 
            batch_ct += 1
            if ((batch_ct + 1) % 400) == 0:
                train_log(loss, example_ct, epoch)
            ########################################################################
        
        #validation
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        total = 0
        # Disable gradient calculation for validation or inference using torch.no_rad()
        #with torch.no_grad():
        for x,y in valid_dl:
            if config['channels_last']:
                x = x.to(config['device'], memory_format=torch.channels_last) #CHW --> #HWC
            else:
                x = x.to(config['device'])
            y = y.to(config['device'])
            valid_logits = model(x)
            _, valid_preds = torch.max(valid_logits, 1)
            valid_loss = criterion(valid_logits,y)
            running_loss += valid_loss.item() * x.size(0)
            running_corrects += torch.sum(valid_preds == y.data)
            total += y.size(0)
            
        epoch_loss = running_loss / len(valid_data)
        epoch_acc = running_corrects.double() / len(valid_data)
        train_acc = run_corrects.double() / len(train_data)
        print("Train Accuracy",train_acc.cpu())
        print("Validation Loss is {}".format(epoch_loss))
        print("Validation Accuracy is {}\n".format(epoch_acc.cpu()))
        if epoch_acc.cpu()>best_acc:
            print('One of the best validation accuracy found.\n')
            #torch.save(model.state_dict(), config['best_saved_path'])
            best_acc = epoch_acc.cpu()

            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    #torch.save(model.state_dict(), config['saved_path'])

    
def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


# In[ ]:


clf = clf.to(config['device'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf.parameters(),lr=config['lr'])
train_model(clf,criterion,optimizer,train_dl,valid_dl,num_epochs=10)


# In[ ]:


class AdversarialDataset(Dataset):
    def __init__(self, dataset, model, epsilon = 0.05):
        self.dataset = dataset
        self.model = model
        self.epsilon = epsilon
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if idx%2:
            return x,0
        x_adv = self.generate_adversarial_example(x, y)
        return x_adv, 1
    
    def generate_adversarial_example(self, x, y):
        x.requires_grad = True
        outputs = self.model(x.unsqueeze(0))
        loss = nn.functional.cross_entropy(outputs, torch.tensor([y]))
        loss.backward()
        x_adv = x + self.epsilon * torch.sign(x.grad)
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv


# In[ ]:


clf = clf.cpu()
#clf.load_state_dict(torch.load('saved_models/random.pt'))

new_train_data = AdversarialDataset(train_data,clf)
new_test_data = AdversarialDataset(test_data,clf)
print(len(new_train_data),len(new_test_data))
new_train_data[0][0].shape, new_train_data[0][1]


# In[ ]:


train_dl1 = torch.utils.data.DataLoader(new_train_data, batch_size=32,shuffle=True)
test_dl1 = torch.utils.data.DataLoader(new_test_data, batch_size=32,shuffle=True)


# In[ ]:


#model = AdversarialDetector()


mobilenet = models.mobilenet_v3_small(pretrained=True)
mobilenet.classifier[3] = nn.Linear(in_features = 1024, out_features = 2, bias = True)
model = mobilenet

model = model.to(config['device'])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=config['lr'])
train_model(model,criterion,optimizer,train_dl1,test_dl1,10)


# In[ ]:


torch.save(model.state_dict(), 'saved_models/attack_detection.pt')

