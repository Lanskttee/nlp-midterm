# -*- coding: UTF-8 -*-
import json
import jsonlines

import nltk
# nltk.download('vader_lexicon')

# from snownlp import SnowNLP
# from snownlp import sentiment
from nltk.sentiment import SentimentIntensityAnalyzer
import operator
# from flair.models import TextClassifier
# from flair.data import Sentence
import numpy as np
# 导包
import collections
import os
import random
import time
from tqdm import tqdm
import torch
from torch import nn
# torchtext的版本应用为0.4.0
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn.functional as F


import re

from nltk.corpus import stopwords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def text_filter(text):

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()
    # In Python, searching a set is much faster than searching
    stops = set(stopwords.words("english"))
    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    # join the words back into one string separated by space, and return the result.
    return( " ".join(meaningful_words))

def process_data():
    decoder = json.JSONDecoder()
    data=[]
    label=[]
    with open('train_dataset.json','r') as fin:
        str_ = fin.readlines()
        # print(eval(str_[0]+str_[1]+str_[2]+str_[3]+str_[4])['content'])
        len_str_=int(len(str_)/5)
        # print(len_str_)

        for i in range(len_str_):
            dict_=eval(str_[0+5*i]+str_[1+5*i]+str_[2+5*i]+str_[3+5*i]+str_[4+5*i])
            # json_res = decoder.raw_decode(line)[0]
            data.append(text_filter(dict_['content']).split(' '))
            label.append(int(dict_['rating'])-1)
        # for item in jsonlines.Reader(fin):
        #     data.append(item["content"])
        #     label.append(item["rating"])

    # print(data)
    # print(label)
    return data, label



data,label=process_data()
counter=collections.Counter([token for sentence in data for token in sentence])
vocab=Vocab.Vocab(counter,min_freq=10)
# print(len(vocab),vocab) #13932 Vocab()

# print(data[0])
def padding(text):
    max_length=500
    return text[:max_length] if len(text)>max_length else text+[0]*(max_length-len(text))

data=torch.tensor([padding([vocab.stoi[word] for word in words]) for words in data])

# new_data=[]
# for words in data:
#     tmp=[]
#     for word in words:
#         print(word)
#         print(vocab.stoi[word])
#         tmp.append(vocab.stoi[word])
#     new_data.append(padding(tmp))
# data=torch.tensor(new_data)
label=torch.tensor(label)


class BiRNN(nn.Module):
    def __init__(self, vocab_len, embed_size, num_hiddens, num_layers):
        '''
        @params:
            vocab: 在数据集上创建的词典，用于获取词典大小
            embed_size: 嵌入维度大小
            num_hiddens: 隐藏状态维度大小
            num_layers: 隐藏层个数
        '''
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_len, embed_size)  # 映射长度,这里是降维度的作用

        # encoder-decoder framework
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)  # 双向循环网络
        self.decoder = nn.Linear(4 * num_hiddens, 5)  # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # 循环神经网络最后的隐藏状态可以用来表示一句话

    def forward(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, seq_len) 的整数张量
        @return:
            outs: 对文本情感的预测，形状为 (batch_size, 2) 的张量
        '''
        # 因为LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置,注意这里转置了!!!!
        embeddings = self.embedding(inputs.permute(1, 0))  # (seq_len, batch_size, d)500*64*100
        # print(embeddings.shape)
        # rnn.LSTM 返回输出、隐藏状态和记忆单元，格式如 outputs, (h, c)
        outputs, _ = self.encoder(embeddings)  # (seq_len, batch_size, 2*h)每一个输出,然后将第一次输出和最后一次输出拼接
        # print(outputs.shape)# 如果是双向LSTM，每个time step的输出h = [h正向, h逆向] (同一个time step的正向和逆向的h连接起来)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)  # (batch_size, 4*h)
        outs = self.decoder(encoding)  # (batch_size, 5)
        return outs


embed_size, num_hiddens, num_layers = 100, 100, 2
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
# net.load_state_dict(torch.load('./model7_33.pth')['model'])


def save(epoch):

    stats = {
        'epoch': epoch,
        'model': net.state_dict()
    }

    savepath =  'model4_{}.pth'.format(epoch+1)
    torch.save(stats, savepath)
    print('saving checkpoint in {}'.format(savepath))



batch_size=32
train_dataset=Data.TensorDataset(data,label)
train_loader=Data.DataLoader(train_dataset,batch_size,shuffle=True)



def train(train_iter,  net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in tqdm(range(num_epochs)):
        if (epoch+1) % 10 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            # print(y_hat)
            # print(y)
            l = loss(y_hat, y) # 交叉熵损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()# 优化方法
            train_l_sum += l.cpu().item()# 进入cpu中统计
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        # test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f,  time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
        if (epoch+1) %5==0:
            save(epoch)

lr, num_epochs = 0.01, 500
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()

train(train_loader, net, loss, optimizer, device, num_epochs)

