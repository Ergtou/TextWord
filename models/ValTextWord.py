# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 13                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 128                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        #self.convs = nn.ModuleList(
        #    [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        #self.embedding.weight.requires_grad = False
        self.dropout = nn.Dropout(config.dropout)
        self.input_dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(config.embed, 2*config.embed)
        #self.fc2 = nn.Linear(2*config.embed, 256)
        self.bifc = nn.Linear(2*config.embed, 2*config.embed)
        #self.trifc = nn.Linear(3*config.embed, 2*config.embed)
        self.fc = nn.Linear(8*config.embed, config.num_classes)

        self.txt1 = nn.Linear(config.embed, 2*config.embed)
        self.txt2 = nn.Linear(2*config.embed, config.num_classes)

    def forward(self, x):
        #out2 = self.embedding(x[2])
        out = self.embedding(x[0])
        out = self.input_dropout(out)
        out = out.mean(dim=1)
        out = self.txt1(out)
        out = F.relu(out)
        out = self.txt2(out)
        
        return out
