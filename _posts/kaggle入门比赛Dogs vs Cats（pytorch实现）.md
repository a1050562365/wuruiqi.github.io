@[TOC]
## 1. 比赛简介
Dogs vs Cats 是kaggle中一个关于图像的入门级比赛，该比赛提供的数据集中包含猫和狗的图片各12500张，总共25000张图片。比赛要求在这个数据集上训练，使计算机能尽可能准确的识别是猫还是狗
## 2. 数据集处理
该数据集较大，我在整个数据集中随机的抽出了猫和狗各500张图片作为测试集，整个数据集的结构如下
- DataSets
  - Train
    - Cats：12000张
    - Dogs：12000张
  - Test
    - Cats：500张
    - Dogs：500张

如果kaggle官网下载数据集过慢的话可以点击[这里](https://www.microsoft.com/en-us/download/details.aspx?id=54765)下载数据集
## 3. 代码实现
#### 3.1 加载处理数据集
kaggle官方所提供的数据集都是jpg格式，而且照片的形状大小并不相同，所以要对数据集进行处理。

**data_load.py**

```python
import torch
from torchvision import datasets, transforms
from torch.utils import data
import torchvision
import os
import matplotlib.pyplot as plt

data_dir = "DataSets"
data_transform = {x: transforms.Compose([transforms.Scale([224, 224]),
                                         transforms.ToTensor()])
                  for x in ["train", "test"]}  # 将图片统一处理为224*224大小并变为tensor
image_dataSets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                                          transform=data_transform[x])
                  for x in ["train", "test"]}
data_loader = {x: torch.utils.data.DataLoader(dataset=image_dataSets[x],
                                              batch_size=16,
                                              shuffle=True)
               for x in ["train", "test"]}

# 返回处理之后的数据集
def dataLoader():
    return data_loader
```

现在我们已经将数据集中的jpg格式图片处理成了torch可以计算的tensor格式，并且每个图片都用标记0或1代表猫或狗。
如果想查看加载的图片，可以添加以下代码：

```python
x_example,y_example = next(iter(data_loader["train"]))
print(y_example)
img = torchvision.utils.make_grid(x_example)
img = img.numpy().transpose([1,2,0])
plt.imshow(img)
plt.show()
```
#### 3.2 模型的建立、训练、测试
##### 模型建立
本次的模型使用vgg16网络，由于vgg16的输出有1000个而我们的输出只有2个，所以我们需要改动该网络的全连接层
另外pytorch为我们专门提供了训练好的vgg16网络，以下代码可以自动下载vgg16网络，如果下载速度较慢可以到清华源下载
```python
model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Linear(7*7*512, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 2),
)
# 取消平均池化层可以提升训练效果，有的地方下载的vgg16网络没有这一层
model.avgpool = torch.nn.Sequential() 
model = model.cuda() # 在gpu上运行该模型，没有gpu的电脑自行调整
print(model)
```
print(model)输出该网络的结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200223224804930.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20200223224820592.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
##### 模型训练
本次模型训练的损失函数为pytorch提供的CrossEntropyLoss函数，使用Adam梯度下降法进行优化，学习率为0.00001
```python
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)
epoch_n = 5
data_loader = dataLoader()

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch + 1, epoch_n))
    print("=" * 25)

    running_loss = 0.0
    running_corrects = 0
    for batch, data in enumerate(data_loader["train"], 1):
        x, y = data
        x, y = Variable(x.cuda()), Variable(y.cuda())

        y_pred = model(x)

        _, pred = torch.max(y_pred.data, 1)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(pred == y.data)

        if batch % 250 == 0:
            print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.2f}%"
                  .format(batch,
                          running_loss / batch,
                          100 * running_corrects / (16 * batch)))
    epoch_loss = running_loss * 16 / len(data_load.image_dataSets["train"])
    epoch_acc = 100 * running_corrects / len(data_load.image_dataSets["train"])
    print("Train: Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss,epoch_acc))

torch.save(model,"DCNet.pth") # 保存模型
```
训练模型的时候我只在训练完一个epoch的时候进行了截图，也是能看到训练的过程是不错的
应该将`epoch_acc = 100 * running_corrects / len(data_load.image_dataSets["train"])`这句话改为`epoch_acc = 100 * float(running_corrects) / len(data_load.image_dataSets["test"])`，不然结果都是整数，像下图这样子
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200223225404962.png)
##### 模型测试
在我们准备好的测试集上测试我们的模型：

```python
print("Testing...")
print("="*25)
running_loss = 0.0
running_corrects = 0
for batch, data in enumerate(data_loader["test"], 1):
    x, y = data
    x, y = Variable(x.cuda()), Variable(y.cuda())

    y_pred = model(x)

    _, pred = torch.max(y_pred.data, 1)
    loss = loss_fn(y_pred, y)

    running_loss += loss.item()
    running_corrects += torch.sum(pred == y.data)

epoch_loss = running_loss * 16 / len(data_load.image_dataSets["test"])
epoch_acc = 100 * float(running_corrects) / len(data_load.image_dataSets["test"])
print("Test: Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss,epoch_acc))
```
经过5次epoch后得到的测试数据如下（准确率应该有点问题，直接取整了）
![!\[在这里插入图片描述\](https://img-blog.csdnimg.cn/20200223225730722.png](https://img-blog.csdnimg.cn/20200223230143388.png)
完整代码如下：
##### model.py(完整代码)

```python
import torch
from torchvision import models
import data_load
from data_load import dataLoader
from torch.autograd import Variable

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Linear(7*7*512, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Dropout(p=0.5),
    torch.nn.Linear(4096, 2),
)
model.avgpool = torch.nn.Sequential()
model = model.cuda()
print(model)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)
epoch_n = 5
data_loader = dataLoader()

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch + 1, epoch_n))
    print("=" * 25)

    running_loss = 0.0
    running_corrects = 0
    for batch, data in enumerate(data_loader["train"], 1):
        x, y = data
        x, y = Variable(x.cuda()), Variable(y.cuda())

        y_pred = model(x)

        _, pred = torch.max(y_pred.data, 1)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_corrects += torch.sum(pred == y.data)

        if batch % 250 == 0:
            print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.2f}%"
                  .format(batch,
                          running_loss / batch,
                          100 * running_corrects / (16 * batch)))
    epoch_loss = running_loss * 16 / len(data_load.image_dataSets["train"])
    epoch_acc = 100 * float(running_corrects) / len(data_load.image_dataSets["test"])
    print("Train: Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss,epoch_acc))

torch.save(model,"DCNet.pth")


print("Testing...")
print("="*25)
running_loss = 0.0
running_corrects = 0
for batch, data in enumerate(data_loader["test"], 1):
    x, y = data
    x, y = Variable(x.cuda()), Variable(y.cuda())

    y_pred = model(x)

    _, pred = torch.max(y_pred.data, 1)
    loss = loss_fn(y_pred, y)

    running_loss += loss.item()
    running_corrects += torch.sum(pred == y.data)

epoch_loss = running_loss * 16 / len(data_load.image_dataSets["test"])
epoch_acc = 100 * float(running_corrects) / len(data_load.image_dataSets["test"])
print("Test: Loss:{:.4f} Acc:{:.4f}%".format(epoch_loss,epoch_acc))
```
#### 3.3 使用模型
在上一步中，我们已经将训练好的模型保存下来，现在我们来直接使用它并判断自己的图片是猫还是狗。
##### test.py

```python
import torch
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

model = torch.load("DCNet.pth")
image_path = "test_image/test.png"


img = Image.open(image_path)
data_transform = transforms.Compose([transforms.Scale([224, 224]),
                                     transforms.ToTensor()])  # 将图片统一处理为64*64大小并变为tensor
img = data_transform(img)
x = []
for i in range(16):
    x.append(img)

x = torch.stack(x,dim=0)
x = Variable(x.cuda())
y = model(x)
y = y[0]
if y[0] < y[1]:
    print("this is a dog")
else:
    print("this is a cat")


img = img.numpy().transpose([1, 2, 0])
plt.imshow(img)
plt.show()

```
##### 测试结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200223231044969.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)