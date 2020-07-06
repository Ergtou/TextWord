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
        self.pad_size = 256                                              # 每句话处理成的长度(短填长切)
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

        self.att = nn.Embedding(config.num_classes, config.embed)
        self.bias = nn.Parameter(torch.Tensor(config.num_classes))

        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform(self.att.weight.data)
        nn.init.normal(self.bias)
    
    def __attention_train_logit__(self, x, query):
        # x-(B,L,H)
        current_relation = self.att(query)
        current_relation = current_relation.unsqueeze(1)
        current_relation = current_relation.expand(x.shape)
        
        #(B,L)
        attention_logit=torch.sum( x * current_relation ,2 ,True)
        return attention_logit

    def __attention_test_logit__(self, x):
        # x-(B,L,H)
        attention_logit = torch.matmul(x ,torch.transpose(self.att.weight,0,1))
        # logit-(B,L,C)
        return attention_logit

    def __logit__(self, x):
        attention_logit = torch.matmul(x ,torch.transpose(self.att.weight,0,1),)
        return attention_logit + self.bias

    def forward(self, x, test=False):
        out2 = self.embedding(x[2])
        #out3 = self.embedding(x[3])
        out = self.embedding(x[0])
        label = x[3]
        
        #out3 = torch.cat([out, out2, out3],2)
        #out = torch.cat([out,out2],2)
        

        #out = self.input_dropout(out)
        #out = self.fc1(out)
        #out = F.relu(out)
        if not test:
        #if False:
            '''
            att_logits = self.__attention_train_logit__(out,label)
            att_scores = F.softmax(att_logits,1).expand(out.shape)
            # (B,H)
            out = att_scores.mul(out).sum(1)
            out_logits = self.__logit__(out)
            '''
            att_logits = self.__attention_train_logit__(out,label)
            att_scores = F.softmax(att_logits,1)
            batch_rep=[]
            for i in range(out.shape[0]):
                # (B,H)
                ins_rep = torch.matmul(torch.transpose(att_scores[i],0,1), out[i])
                ins_repre = torch.squeeze(ins_rep)
                batch_rep.append(ins_repre)
            bag_repre = torch.stack(batch_rep)
            #logits = self.classifier(self.drop(bag_repre))
            #bag_repre = self.dropout(bag_repre)
            out_logits = self.__logit__(bag_repre)
        else:
            batch_logit = []
            for i in range(out.shape[0]):
                #(L,C)
                att_logits = self.__attention_test_logit__(out[i])
                #(L,C)
                att_scores = F.softmax(att_logits,0)
                #(C,L)
                att_scores = torch.transpose(att_scores,0,1)
                #(C,L,H)
                #att_scores = att_scores.unsqueeze(3).expand([att_scores.shape[0],att_scores.shape[1],att_scores.shape[2],out.shape[2]])
                rep = torch.matmul(att_scores,out[i])
                #(B,C,L,H)
                #out = out.unsqueeze(1).expand(att_scores.shape)
                #(B,C,H)
                #out = att_scores.mul(out).sum(2)
                #(B,C,C)
                out_logits = self.__logit__(rep)
            
                batch_logit.append(torch.diag(F.softmax(out_logits,1)))
            #(B,C)
            out_logits = torch.stack(batch_logit)

        #out2 = self.input_dropout(out2)
        #out2 = self.bifc(out2)
        #out2 = F.relu(out2)
        
        #out = torch.cat([out.max(dim=1)[0],out.mean(dim=1)],1)

        

        #out2 = torch.cat([out2.max(dim=1)[0],out2.mean(dim=1)],1)
        #out = torch.cat([out,out2],1)
        #out = self.fc(out)
        
        return out_logits
