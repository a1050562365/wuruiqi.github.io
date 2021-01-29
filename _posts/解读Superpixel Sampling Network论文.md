@[toc]
## 1. 介绍
### 1.1 超像素概念
超像素最直观的解释，就是把一些具有相似特性的像素“聚合”起来，形成一个更具有代表性的大“元素”。对于较大的图片，可以在不牺牲太大精度的前提下对图片进行“降维”。
如下图，每个小块代表着具有将具有相似特征的像素聚合得到的元素
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715182328799.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
### 1.2 超像素与深度学习
随着深度学习算法的流行，越来越多的任务可以通过神经网络来实现，并且在算法结果、运算效率等方面都有很好的表现，但是由于以下两个原因，深度学习并没有很好的和超像素分割任务相结合：
1. 形成大多数深层结构基础的标准卷积运算通常在规则网格上定义，并且当在不规则超像素晶格上操作时变得低效。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715183342193.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
从图中我们看到，卷积运算适合在用像素这种规则网格表示的图片上进行运算，而超像素的形状是不规则的，难以在其上进行卷积运算。（自己的理解，应该是这个意思，不过本文并没有重点解决该问题）
2. 现有的超像素算法基于最邻近运算，是不可微分的，因此在深度网络中使用超像素使得在端到端的可训练网络架构中引入了不可微分的模块。
在神经网络训练的过程中，需要使用到反向传播算法，如果引入了不可微分的模块，则神经网络难以训练。
### 1.3 本文的贡献
- 提出了可微分的超像素算法，解决了上述的第二个问题。
- 利用该算法提出端到端的SSN网络，且可以灵活地用在各种任务上。
- 超像素分割算法的SOAT，令人满意的运行时间。

## 2. 前提知识
### 2.1 SLIC算法简介
SSN方法的核心是可微分的超像素聚类算法，本文在SLIC算法的基础上进行修改，得到了可微分的超像素算法（Differentiable SLIC），所以在了解SSN之前我们先来了解一下SLIC算法，
SLIC算法的本质就是一种k-means算法，将每个像素点用它的XYLAB信息（坐标信息、LAB颜色空间信息）表示为一个五维向量。并在所有的向量的集合上进行k-means聚类。具体的流程如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715184824805.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
其中 Si 代表第i个超像素元素的聚类中心，Ip 代表第 P 个像素点，Hp 代表第 P 个像素点所属的超像素元素，Zi 代表第i个超像素元素中所包含的像素数量。上标 t 代表第 t 次迭代
该算法主要分如下两步：
1. 像素-超像素关联：通过计算每个像素和超像素中心之间的距离（其中D代表两个向量的欧式距离的平方），找到最小值，即距离最近的超像素，来确定该像素所属的超像素。
2. 超像素中心更新：对每个超像素簇中所有的像素的特征进行平均，获得簇中心，进而得到这次迭代后的聚类中心。

反复执行上述两个步骤，得到超像素分割的最终结果，就是SLIC算法的核心流程。
## 3. SSN
SSN由两部分组成，一是用来提取图像信息的神经网络，而是将神经网络提取的信息用可微分的SLIC算法进行计算。
### 3.1 可微分的SLIC
首先我们可以看到，在SLIC算法中存在不可微分的计算，即在像素-超像素关联过程中的最邻近运算。于是，本文在此基础上进行了改进，提出了可微的SLIC算法。
1. 在像素-超像素关联这一步中，提出了通过计算权重的方式来代替最邻近运算。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715205437257.png)
该公式表示第 t 次迭代过程中，计算像素 p 和超像素 i 的权重。
2. 更新超像素中心：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715205616596.png)
 这里通过加权的方式更新超像素中心，其中Z代表所有像素对超像素 i 的权重的和，最终我们可以得到这个式子：![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715205818348.png)
 其中Q_hat代表Q矩阵的列归一化矩阵。
 3. 细节：为了节省计算时间，只计算像素与周围九个超像素之间的权值，大大降低了算法的时间复杂度。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715211141373.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
 ### 3.2 SSN网络及代码实现
 该网络图像如下![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715210234597.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
 其中 ↑ 符号代表双线性插值上采样
 ### 3.3 算法步骤
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715211345578.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
### 3.4 像素和超像素表示之间的映射
1. 像素→超像素：
从之前描述的可微SLIC算法过程中我们已经得到了像素到超像素的映射。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715214547783.png)
2. 超像素→像素：
传统方法中，一般通过将超像素的特征分配给在该超像素中的所有像素。
而在该算法中，可以通过乘以行归一化的Q来计算，通过公式表示如下（该公式个人理解就是通过对每个超像素做加权，类似于之前公式中对每个像素点做加权来更新超像素中心）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715214602725.png)
其中F表示像素，S表示超像素
## 4. 学习特定任务的超像素
灵活是SSN的优点之一，它可以与任何特定于任务的损失函数相结合，从而学习有利于得到特定任务信息的超像素表示。
### 4.1 特定任务的重建损失
假设我们可以在特定任务中想要有效表示的像素属性为R（如语义标签、光流图）
通过 像素→超像素 的映射 再到  超像素→像素 的逆映射，得到经过超像素分割后表达的像素信息R*，通过R和R*建立起损失函数如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715221758728.png)
其中L为特定任务的损失函数。
### 4.2 紧凑性损失
除了上面的损失，也使用了一个紧凑性损失来鼓励超像素实现空间上的紧凑性。也就是在超像素簇内部有着更低的空间方差。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715221859297.png)
使用L2损失函数，Ixy代表位置特征。
### 4.3 损失函数
最终的损失函数为上述两个损失函数的组合：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715222007464.png)
其中lambda的值为1e-5
## 5 实验细节
### 5.1 评估指标
1. ASA（Achievable Segmentation Accuracy）表示在超像素的基础上执行分割的准确度的上限。
2. 边界召回（BR）和边界精度（BP）指标测量超像素边界与GroundTruth的对齐程度。

### 5.2 消融实验
作者将算法分为三类：
1. SSNpix，将XYLab特征作为输入，与SLIC算法类似，不经过神经网络的处理。
2. SSNlinear，将卷积网络替换为一个简单的卷积层
3. SSNdeep，即为作者提出的包含7个卷积层的SSN
并通过设定神经网络提取不同的特征数 k 和算法迭代次数 v 来进行消融实验。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715225029944.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
从图中可以看到随着神经网络的层数、k、v的增加，算法的ASA和BR得分也在增加。
### 5.3 比较
规定v = 10, k = 20后进行比较：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200715225300204.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/202007152253088.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg0OTc2Mw==,size_16,color_FFFFFF,t_70)
通过与其它算法的比较，我们可以看到SSN算法都有明显的优势。
### 5.4 其它
除此之外，作者还针对语意分割、光流图等任务进行了实验和比对，这里不再赘述。
## 6. 总结
提出了一种新颖的超像素采样网络（SSN），它利用通过端到端训练学到的深层特征来估计任务特定的超像素。这是第一个端到端可训练的深度超像素预测技术。

- 实验的几个基准测试表明，SSN始终如一地在表现出色，同时也更快。
- 将SSN集成到语义分割网络中还可以提高性能，显示SSN在下游计算机视觉任务中的实用性。
- SSN快速，易于实施，可以轻松集成到其他深层网络中，具有良好的实证性能。
- SSN解决了将超像素纳入深度网络的主要障碍之一，这是现有超像素算法的不可微分性质。
- 在深度网络中使用超像素可以具有几个优点。
  - 超像素可以降低计算复杂度，尤其是在处理高分辨率图像时
  - 超像素也可用于加强区域常量假设（enforce piece-wise constant assumptions）
  - 有助于远程信息传播
  

相信这项工作开辟了利用深层网络中的超像素的新途径，并激发了使用超像素的新深度学习技术。