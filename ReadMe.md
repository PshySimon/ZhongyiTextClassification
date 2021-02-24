# 运行环境：

+ Linux

+ Python 3.7

## 安装的库有

+ Pytorch 1.7.0

+ Transformers 3.5.1

+ Jupyter Notebook

+ Numpy

+ Sklearn

+ Pandas

+ Matplotlib

+ Seaborn

+ Xlrd

+ W3lib

## 目录结构

D:.

│ ExtractText.ipynb		从数据集中提取文本数据，读取数据放在同级目录的data/train和data/test2目录下

│ prediction.csv			  最终的预测文件

│ Preprocess.ipynb		预处理

│ problem_file.txt		   部分xlrd打不开的文件，有些利用wps做了修正（去除了数据前后的空格），一部分

​										   改成同名的空文件

│ Utils.ipynb					将有问题的文件复制到数据集中

│ 预测结果整合.ipynb	 预测结果整合

├─data

...

└─model

  │ answer_train.csv					带标签的训练集

  │ train.csv									带有文件内容的训练集

  │

  ├─content

  │ │ content_predict.ipynb		利用文件内容进行预测

  │ │ DataIter.py							处理数据成Bert的格式

  │ │ submit_content.csv			 靠文件内容预测的部分

  │ │ test_content_v1.csv			测试集，包含测试文件的内容

  │ │ train_v1.csv						  训练集

  │ │

  │

  └─title

​    │ DataIter.py							处理数据成Bert的格式

​    │ submit_title.csv					靠文件标题预测的部分

​    │ test_title_v1.csv				   测试集，包含测试文件的文件名

​    │ title_predict.ipynb				利用文件名进行预测

​    │ train_v1.csv							训练集

##  运行顺序

+ 先将数据解压到data/train和data/test2目录下
+ 运行Utils.ipynb文件，将通过wps或者改为空文件修正后的文件复制到文件夹下
+ 运行ExtractText.ipynb生成train.csv文件
+ 运行Preprocess.ipynb生成只靠文件名预测和只靠文件内容预测的模型需要的数据
+ 运行title_predict.ipynb
+ 运行content_predict.ipynb

## 说明

​		本次比赛才用的是将文件名和内容分开预测，然后将测试集合并起来的策略。

​		模型使用的是**Ernie1.0 Chinese**，读取Ernie的配置文件分别在**model/title/title_predict.ipynb**和**model/content/content_predict.ipynb**文件的第一个cell中.

​		有一部分文件无法打开，问题列在根目录problem_file.txt文件下，都做了说明。

​		可能有的文件没法打开我忘记说明了，当时我是删掉这个文件又重新新建了一个空白的同名文件。



