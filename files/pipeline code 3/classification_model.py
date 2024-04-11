import torch
import torch.nn as nn 
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import time
from tqdm import tqdm
import pandas as pd

class inference:
    def __init__(self,data_dir='C:/Users/SumeetMitra/dataset/dataset',
                 epoch=10,
                 save_path='weights',
                 learning_rate=0.01
                 ):
        self.train_dir = data_dir+'/train'
        self.valid_dir = data_dir+'/valid'
        self.epoch = epoch
        self.save_path = save_path
        self.learning_rate = learning_rate
        self.read_folder()
        self.read_model()
        self.read_optim()
        self.train_model()
        
        
    def read_folder(self):
        transform = transforms.Compose([transforms.Resize((64,64),
                                        transforms.InterpolationMode.BILINEAR),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                        ])
        train = ImageFolder(self.train_dir,transform)
        valid = ImageFolder(self.valid_dir,transform)
        self.train_data = torch.utils.data.DataLoader(train,shuffle=True,batch_size=8,num_workers=4)
        self.valid_data = torch.utils.data.DataLoader(valid,shuffle=True,batch_size=8,num_workers=4)
        self.classes = train.class_to_idx
        
    def read_model(self):
        out_feature = len(self.classes)
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=out_feature,bias=True)
        self.model = model
        
    def read_optim(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer =fit_optimizer =  optim.SGD(self.model.parameters(),lr=self.learning_rate,momentum=0.9)
        self.learning_rate_scheduler = lr_scheduler.StepLR(fit_optimizer,step_size=2, gamma=0.05)
        
        
    def train_model(self):
        lst = []
        since = time.time()
        best_accuracy = 0.0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("The model will be running on", device, "device")
        self.model.to(device)
        self.model.train()
        for epoch in range(self.epoch):
            print('Epoch {}/{}'.format(epoch, self.epoch))
            running_loss = 0.0
            accuracy = 0.0
            for (images,labels) in tqdm(self.train_data):
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                loss.backward()
                self.optimizer.step()
                total += labels.size(0)
                running_loss += loss.item()
                accuracy += (predicted == labels).sum().item()
            accuracy = (accuracy / total)
            print('{} Loss: {:.4f} Acc: {:.4f} %'.format(epoch, running_loss, accuracy))
            val_accuracy = self.valid_model()
            lst.append([epoch,since,running_loss,accuracy,val_accuracy])
            if val_accuracy > best_accuracy:
                self.saveModel('epoch_'+epoch)
                print('Saved model: epoch {:.4f} Acc: {:.4f} %'.format(epoch,val_accuracy))
            pd.DataFrame(lst,columns=['epoch','time','train_loss','train_acc','val_acc'])   
            pd.to_csv('results.csv') 
            
    def valid_model(self):
        model = self.model
        model.eval()
        accuracy = 0.0
        total = 0.0
        with torch.no_grad():
            for data in tqdm(self.valid_data):
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
        accuracy = (accuracy / total)
        print('Validation Accuracy : {:.4f}'.format(accuracy))
        return(accuracy)
    
    def saveModel(self,save_name):
        torch.save(self.model.state_dict(), self.save_path+"/"+str(save_name)+".pt")
                
                
inference()