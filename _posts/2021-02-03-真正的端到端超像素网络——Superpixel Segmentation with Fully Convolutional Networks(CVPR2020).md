---
layout: post
categories: 论文笔记
cover: 'https://img-blog.csdnimg.cn/20210203170801294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70'
---

## 0. 传送门
论文地址：[https://arxiv.org/abs/2003.12929](https://arxiv.org/abs/2003.12929)

github地址：[https://github.com/fuy34/superpixel_fcn](https://github.com/fuy34/superpixel_fcn)
## 1. 简介
超像素最直观的解释，就是把一些具有相似特性的像素“聚合”起来，形成一个更具有代表性的大“元素”。

目前超像素难以和深度学习相结合主要由两个原因：
 - 标准卷积运算是在规则网格上定义的，当应用于超像素时其效率会变的很低。
 - 超像素的聚类操作具有不可微分的计算，使得无法使用反向传播进行深度学习。
 
在我之前的[解读Superpixel Sampling Network论文](https://blog.csdn.net/weixin_43849763/article/details/107367652)的博客中，介绍的SSN就是解决了第二个问题，将argmax操作用softmax替代，使得计算可微。

这篇Superpixel Segmentation with Fully Convolutional Networks则是通过一个巧妙的转化，将超像素分割作为一个深度学习任务，解决了第一个问题。他可以直接得到超像素分割结果，不需要额外的聚类操作。

该论文的贡献点主要如下：
 - 使全卷积网络可以快速生成超像素，省去了聚类的步骤，一个encoder-decoder可以说是最简单的神经网络结构，颇有大道至简的味道。
 - 提出了一个上采样/下采样框架，使得超像素能很好的和其他任务结合，文章中也给出了超像素应用于stereo matching的pipeline。

## 2. 方法
#### 2.1 在规则网格上学习超像素
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203165550351.png)
大多数的超像素方法都会在图片上规则地初始化超像素中心，通过计算每个像素和超像素的关联得到一个$H \times W \times N_S$的矩阵，其中$N_S$为超像素的数量。

但是逐超像素的计算太复杂，通过观察不难发现像素基本只会被划分为附近的超像素中，所以可以只考虑周围的超像素，比如图中绿色框的像素只需要考虑与红色框中9个超像素的关联。这样我们最后只需要得到一个$H \times W \times 9$的矩阵$Q$。时间复杂度从$O(N \times N_S)$减少到$O(N)$。

SpixelFCN也就是通过神经网络预测这个$Q$从而得到超像素的结果。

#### 2.2 SpixelFCN VS SSN
如果看过SSN，应该会发现这个$Q$在SSN中也出现过，不过SSN是用卷积网络提取图片的深度特征，然后送到一个软聚类模块中，计算得到$Q$，由于图像动辄几十万的像素点，导致聚类花费了大量的时间。

然而SpixelFCN直接用卷积网络预测$Q$，省去了聚类的时间，使得超像素分割的效率大大提升。网络结构和与SSN的对比图如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203170801294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)


![一步到位，大道至简](https://img-blog.csdnimg.cn/20210203162827779.png)
#### 2.3 方法细节（啃公式）
1. 关联矩阵 $\to$ 超像素中心信息
我们将超像素中心$s$表示为$C_s=(u_s,I_s)$，其中$u_s$代表特征信息，$I_s$代表位置信息，我们可以按如下公式计算超像素中心：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203171304349.png)
$f(p)$是像素$P$的特征，$q_s(P)$是$Q$中像素$P$与超像素$s$的关联，$N_P$是像素周围的超像素集合。
整个公式就是按照像素与超像素的关联程度，以加权平均的方式求得超像素中心的特征。
2. 超像素中心 $\to$ 重建图像
![在这里插入图片描述](https://img-blog.csdnimg.cn/2021020323102557.png)
利用像素和超像素的关联，在超像素特征的基础上重建图像。
3. 损失函数
损失函数的设计上与SSN类似
    - 一般形式$L(Q)$：![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203231903983.png)
重建后的特征与原特征的差距（前后相似性）+重建后的位置与原位置的差距（空间紧凑性）
   - 类似SLIC的形式$L_{SLIC}(Q)$:
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/202102032321347.png)
将特征用CIELAB表示，L2模计算dist，类似于SLIC中的做法。
    - 以语义标签为基础的形式$L_{sem}(Q)$:
   ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203232320286.png)
特征用one-hot形式的语义标签表示，使用cross-entropy计算dist，从而获得更贴合语义边缘的超像素。

4. CSP表示超像素中心
其实也没看懂花里胡哨干了什么，只是换了一个计算的方式，可能这么计算会更快吧。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203232642365.png)
本来CSP形式是这样的，然后作者进行了改写如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203232723848.png)
看着很复杂，对这个公式进行一个简单的推理，其实和公式(1)是等价的，推导过程如下（手写的，markdown敲这玩意太麻烦了，字丑莫怪）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203232834105.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
## 3. 实验结果
介绍完了方法之后，我们来看看华丽丽的实验结果。
首先是几个标准下的对比，作者选用的是ASA，BR/BP，CO
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203233232692.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
取得了SOTA的效果，有的地方比SSN差一点，总体还是不错的。

但是但是但是，SpixelFCN的速度和其他的deep-learning-based方法相比，快了一个数量级！
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210203233416422.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
这张图不得不说，挺震撼的哈哈，毕竟只有一个卷积网络，肯定很快，论文上描述可以达到50fps。

## 4. 个人总结

- 优点：端到端的神经网络，将超像素分割转化为预测像素和超像素之间的关联，达到sota的同时大大提升了速度。
- 缺点：准确的说，论文中的网络学到的是$16 \times 16$网格下预测像素和超像素之间的关联，所以对于一个图片，想要获得不同数量的超像素需要将图片resize成不同的大小，如果图片的形状和训练集的形状差别很大，那么效果就会不太好（曾经在Cityscapes上跑过，得分非常不错，但是很多细节无法分割出来，因为对于这么大的图片，resize成网络要求的大小时已经丢失了很多细节了）