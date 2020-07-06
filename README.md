# TextWord
TextWord for chinese short text classifitcation

中文短文本分类，TextWord

## 介绍
原始框架博客：[博客](https://zhuanlan.zhihu.com/p/73176084)  

原始框架Github：[Github地址](https://github.com/649453932/Chinese-Text-Classification-Pytorch) 

## 效果

模型|acc|备注
--|--|--
TextWord|80.4%|使用TextWord,13 epoch
TextCNN|77.2%|Kim 2014 ,13 epoch
FastText|74.6%|bow+bigram+trigram，40 epoch



## 使用说明
```
# 训练并测试：
# TextWord
python run.py --model TextWord --word True
```

## 数据集处理
使用各数据集下的jupyter notebook文件处理各个数据集

处理后重命名该文件夹为THUCNews

下载中文词嵌入文件放到数据集文件夹下

运行util.py文件建立词典和对应词嵌入

## 模型
每个单词和每两个单词过一次全连接  ->  分别取Mean和Max后拼接  ->  Softmax