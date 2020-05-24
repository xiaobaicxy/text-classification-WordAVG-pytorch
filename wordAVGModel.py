# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:55:42 2020

@author: 86186
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_processor import DataProcessor

torch.manual_seed(123) #保证每次运行初始化的随机数相同

vocab_size = 5000   #词表大小
embedding_size = 100   #词向量维度
hidden_size = 128  #神经网络隐藏层大小
num_classes = 2    #二分类
sentence_max_len = 64  #单个句子的长度

lr = 1e-3
batch_size = 16   
epochs = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义模型，句子中个词向量的平均值表示句向量
class WordAVGModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_classes):
        super(WordAVGModel, self).__init__()
        self.liner1 = nn.Linear(embedding_size, hidden_size)
        self.liner2 = nn.Linear(hidden_size, num_classes)
        self.act_func1 = nn.ReLU()
        self.act_func2 = nn.Softmax(dim=1)
    
    def forward(self, x):
        #x [batch_size, sentence_length, embedding_size]
        #用窗口为 句子长度*1 二维平均池操作得到句向量
        x = F.avg_pool2d(x, (x.size(1), 1))  #[batch_size, 1, embedding_size]
        x = x.squeeze(1)   #[batch_size, embedding_size]
        x = self.liner1(x)
        x = self.act_func1(x)
        x = self.liner2(x)
        x = self.act_func2(x)
        return x
def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)
        
        preds = model(datas)
        loss = loss_func(preds, labels)
        
        loss_val += loss.item() * datas.size(0)
        
        #获取预测的最大概率出现的位置
        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)
        corrects += torch.sum(preds == labels).item()
    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    return test_acc

def train(model, train_loader,test_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for datas, labels in train_loader:
            datas = datas.to(device)
            labels = labels.to(device)
            
            preds = model(datas)
            loss = loss_func(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item() * datas.size(0)
            
            #获取预测的最大概率出现的位置
            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            corrects += torch.sum(preds == labels).item()
        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)
        if(epoch % 2 == 0):
            print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
            test_acc = test(model, test_loader, loss_func)
            if(best_val_acc < test_acc):
                best_val_acc = test_acc
                best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model

processor = DataProcessor()
train_datasets, test_datasets = processor.get_datasets(vocab_size=vocab_size, embedding_size=embedding_size, max_len=sentence_max_len)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)


model = WordAVGModel(embedding_size, hidden_size, num_classes)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.BCELoss()
model = train(model, train_loader, test_loader, optimizer, loss_func, epochs)

