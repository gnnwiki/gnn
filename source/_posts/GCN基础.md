---
title: GCN的直观理解
catalog: true
date: 2019-07-23 16:23:22
subtitle:
author: 张尧
header-img:
tags:
- GCN
---
# 图卷积GCN的直观理解
Author: Yao Zhang

实验室又有新生进来了，也有对图卷积感兴趣的，那就写篇文章，简单的讲一下。本篇文章参考了[Kipf的博客](https://tkipf.github.io/graph-convolutional-networks/)，主要是直观的理解，加上我的一点解读，数学上的推导咱们以后有空再说。

<a name="iJgSt"></a>
# 前言

GCN即Graph Convolutional Network，我个人觉得指代了三种东西，第一种是说GCN这个算法，即Kipf and Welling在ICLR'2017发表的[paper](https://arxiv.org/abs/1609.02907)中提出的算法，也是本文介绍的对象；第二种是说基于Spectral Graph Theory的一类图卷积网络，当然Kipf这个模型也是其中一种，此外还有[ChebyNet](https://arxiv.org/abs/1606.09375)等；第三种指代最general，主要指那些基于邻居聚合（Neighborhood Aggregation）思想的定义在图上的神经网络模型，但现在大家更倾向用GNNs（Graph Neural Networks）来概括。

*在[PinSAGE](https://arxiv.org/abs/1806.01973)模型中，作者还用了这样的脚注来说明文中的GCN是指代第三种：
> Following a number of recent works (e.g., [13, 20]) we use the term "convolutional" to refer to a module that aggregates information from a local graph region and to denote the fact that parameters are shared between spatially distinct applications of this module; however, the architecture we employ does not directly approximate a spectral graph convolution (though they are intimately related) [6].


<a name="dMiei"></a>
# 从直觉出发推导GCN

给定一个无向无权图![](https://cdn.nlark.com/yuque/__latex/bd1edafca0a4613576bcc66ce9acaaa0.svg#card=math&code=%5Cmathcal%7BG%3D%28V%2C%20E%29%7D&height=18&width=65)，![](https://cdn.nlark.com/yuque/__latex/89135ae0eb5bcc1af22e8a6d910c39d8.svg#card=math&code=%5Cmathcal%20V&height=13&width=9)是节点的集合，![](https://cdn.nlark.com/yuque/__latex/cca20bbce106d65fbf007dc7b3859a99.svg#card=math&code=%5Cmathcal%7BE%5Csubset%20V%5Ctimes%20V%7D&height=13&width=65)是边的集合。![](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg#card=math&code=A&height=13&width=11)是对称的邻接矩阵，![](https://cdn.nlark.com/yuque/__latex/7fb781eb98fd48687da35391cb09c578.svg#card=math&code=A_%7Bij%7D%3D1&height=18&width=48) iff ![](https://cdn.nlark.com/yuque/__latex/0e58b62dad3c4b8f8e23461635d76731.svg#card=math&code=%28i%2C%20j%29%20%5Cin%20%5Cmathcal%7BE%7D&height=18&width=56)。注意，图中没有自环则对角元素均为0。

GCN的目的是，给定图![](https://cdn.nlark.com/yuque/__latex/bd1edafca0a4613576bcc66ce9acaaa0.svg#card=math&code=%5Cmathcal%7BG%3D%28V%2C%20E%29%7D&height=18&width=65)和所有节点的初始特征构成的特征矩阵![](https://cdn.nlark.com/yuque/__latex/f9299a7f5d76615e7c9d83439ca1cdd9.svg#card=math&code=H%5E%7B%280%29%7D%20%5Cin%20%5CRe%5E%7B%7C%5Cmathcal%7BV%7D%7C%5Ctimes%20d_0%7D&height=18&width=92)，通过一个神经网络输出各个节点的新的特征![](https://cdn.nlark.com/yuque/__latex/a737fd2cca76bb2158ddd6b67c7e2503.svg#card=math&code=H%5E%7B%28L%29%7D%5Cin%20%5CRe%5E%7B%7C%5Cmathcal%7BV%7D%7C%5Ctimes%20d_L%7D&height=18&width=95)，特征维度是否相同都无所谓，方便起见，我们假设神经网络每一层的输入输出维数都相同，因此也有![](https://cdn.nlark.com/yuque/__latex/e92f3369dd8e9b5c955f60764ffeeb62.svg#card=math&code=d%20%3D%20d_0%20%3D%20%5Ccdots%20%3D%20d_L&height=16&width=115)。

为什么我们输入了节点的特征，最后的输出也还是节点的特征呢？如果将每个节点看作一个实体，那么初始特征可能仅反映了实体自身的特征，而通过图卷积，我们希望利用节点之间的连接关系，丰富其特征表示，这样输出的特征![](https://cdn.nlark.com/yuque/__latex/a737fd2cca76bb2158ddd6b67c7e2503.svg#card=math&code=H%5E%7B%28L%29%7D%5Cin%20%5CRe%5E%7B%7C%5Cmathcal%7BV%7D%7C%5Ctimes%20d_L%7D&height=18&width=95)应该除了自身的特征外，还“借鉴”了其他节点的特征，这样表达能力会更强。

<a name="2OdfS"></a>
## Layer-wise Propagation

神经网络的思想就是学习数据的层次化的特征表示。因此，我们自然地想到应该也可以以一种层次化的方式学习节点的特征，即逐层更新节点的特征：<br />![](https://cdn.nlark.com/yuque/__latex/0c5eb42c518e3c4e4969cb89cc14317e.svg#card=math&code=H%5E%7B%28l%2B1%29%7D%20%3D%20f%28H%5E%7B%28l%29%7D%2C%20A%3B%20W%5E%7B%28l%29%7D%29.&height=21&width=164)<br />![](https://cdn.nlark.com/yuque/__latex/8fa14cdd754f91cc6554c9e71929cce7.svg#card=math&code=f&height=16&width=8)是神经网络的一层，拥有参数![](https://cdn.nlark.com/yuque/__latex/734c86fd8d7bdc30951f9ecced92c10c.svg#card=math&code=W%5E%7B%28l%29%7D&height=18&width=28)。我们这里加入了邻接矩阵，是希望特征![](https://cdn.nlark.com/yuque/__latex/0612a6d4b54b338e596b55198df46021.svg#card=math&code=H%5E%7B%28l%29&height=18&width=26)在传播时，考虑到节点之间的连接信息。那么![](https://cdn.nlark.com/yuque/__latex/8fa14cdd754f91cc6554c9e71929cce7.svg#card=math&code=f&height=16&width=8)应该怎么设计呢？这里就要引出最重要的一个思想：Neighborhood Aggregation。<br />

<a name="zf7DJ"></a>
## Neighborhood Aggregation

Neighborhood Aggregation的思想其实特别简单，就是一个节点![](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg#card=math&code=u&height=10&width=8)可以由它的邻居来体现![](https://cdn.nlark.com/yuque/__latex/23fbad0b76c3b2886b8395fdb3dccc1f.svg#card=math&code=N%28u%29&height=18&width=33)，因此，节点的特征就由它的“邻居”的特征“聚合”而成。众多的图卷积算法其实就在“邻居”和“聚合”操作的定义上有所不同。

回到本篇的主题，我们来试着定义一下邻居和聚合操作。最简单的邻居当然就是真的邻居，即与中心节点![](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg#card=math&code=u&height=10&width=8)直接相连的一阶邻居![](https://cdn.nlark.com/yuque/__latex/3cb8158a31edb3d99b1841ac96369ab1.svg#card=math&code=N%28u%29%20%3D%20%5C%7Bv%20%7C%20%28u%2C%20v%29%20%5Cin%20%5Cmathcal%7BE%7D%20%5C%7D&height=18&width=140)，而最简单的聚合就是求和。因此我们有<br />![](https://cdn.nlark.com/yuque/__latex/5e42f234f3f55a629d53d3222a334bb3.svg#card=math&code=%5Ctilde%7Bh%7D_u%5E%7B%28l%2B1%29%7D%20%3D%20%5Csum_%7Bv%20%5Cin%20N%28u%29%7D%20h_v%5E%7B%28l%29%7D.%20%5Ctag%7B1%7D&height=41&width=643)<br />当然，为了进一步增加表达能力，我们会进行非线性变换：![](https://cdn.nlark.com/yuque/__latex/388ef4e06b5039f66286b4bacdd43c47.svg#card=math&code=h_u%5E%7B%28l%2B1%29%7D%20%3D%20%5Csigma%28%5Ctilde%7Bh%7D_u%5E%7B%28l%2B1%29%7D%20%5Ccdot%20W%5E%7B%28l%29%7D%29&height=24&width=149)，其中![](https://cdn.nlark.com/yuque/__latex/a2ab7d71a0f07f388ff823293c147d21.svg#card=math&code=%5Csigma&height=10&width=8)是非线性激活函数，![](https://cdn.nlark.com/yuque/__latex/546de24e1756b0c56e9c8fb45d63f1e1.svg#card=math&code=W%5E%7B%28l%29%7D%5Cin%20%5CRe%5E%7Bd%20%5Ctimes%20d%7D&height=18&width=80)是参数矩阵。把这两个式子写成矩阵形式就是<br />![](https://cdn.nlark.com/yuque/__latex/d8da67d5657e0eb342495f352766a5cf.svg#card=math&code=H%5E%7B%28l%2B1%29%7D%20%3D%20%5Csigma%5Cleft%28%20AH%5E%7B%28l%29%7DW%5E%7B%28l%29%7D%5Cright%29.%20%5Ctag%7B2%7D&height=31&width=643)<br />大家可以自己验证一下这个式子。<br />

<a name="peSNB"></a>
## Self-connection and Normalization

上面讲了Neighborhood Aggregation的思想，得到了一个逐层传播的公式(2)，但这个公式还有两个小问题需要解决一下。

首先，邻接矩阵的对角元素为0，因此我们在应用公式(2)的时候，节点自身的信息会被完全抛弃掉，从式(1)也能看出来，中心节点![](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg#card=math&code=u&height=10&width=8)的新特征与其旧特征完全无关。这可不行，我们得加回来，最直接的方法就是为每个节点加一个自连接（Self-connection），因此我们可以用新的邻接矩阵来代入到上面的公式：![](https://cdn.nlark.com/yuque/__latex/aa85e032d575200688d0ecdb14549f22.svg#card=math&code=%5Ctilde%7BA%7D%20%3D%20A%20%2B%20I&height=18&width=68)，即原来的邻接矩阵加上对角阵。我们通过简单的加入自连接就解决了自身信息丢失的问题。

另一个问题是，节点的特征是通过对邻居节点的特征相加得到的，这样特征的量级会越来越大，因此需要进行归一化（Normalization）。最直观的方法，我们用均值代替式(1)(2)中的求和就好，写成矩阵形式：<br />![](https://cdn.nlark.com/yuque/__latex/10e894f0bc4d253ee9f101fa615de10a.svg#card=math&code=H%5E%7B%28l%2B1%29%7D%20%3D%20%5Csigma%5Cleft%28%5Ctilde%7BD%7D%5E%7B-1%7D%5Ctilde%7BA%7DH%5E%7B%28l%29%7DW%5E%7B%28l%29%7D%20%5Cright%29.%20%20%20%20%20%20%5Ctag%7B3%7D&height=31&width=643)<br />这里，![](https://cdn.nlark.com/yuque/__latex/6404f5b0287c502e4de0c838d92856cc.svg#card=math&code=%5Ctilde%7BD%7D&height=16&width=12)是由节点的度数![](https://cdn.nlark.com/yuque/__latex/89594b1149bcc415cbd1cdc52e4ac0c5.svg#card=math&code=%5Ctilde%7BD%7D_%7Buu%7D%20%3D%20%5Csum_v%20%5Ctilde%7BA%7D_%7Buv%7D%5C%20%3D%20%5Ctilde%7Bd%7D_u&height=35&width=135)构成的对角阵，注意，是加入了自连接后的图。你看，如果我们一开始假设邻接矩阵的对角元素为1，就不用每个符号上面多带一个～了。

式(3)确实解决了数量级爆炸的问题，但不够“漂亮”。原本前面乘的![](https://cdn.nlark.com/yuque/__latex/554e7046abe411a2f1b0a29b1d3d21b8.svg#card=math&code=%5Ctilde%7BA%7D&height=16&width=11)还是一个对称阵，结果现在变成了![](https://cdn.nlark.com/yuque/__latex/1eb6819666a37cf149e5621a87f73034.svg#card=math&code=%5Ctilde%7BD%7D%5E%7B-1%7D%5Ctilde%7BA%7D&height=20&width=38)，不对称了，这可不行。因此，我们可以引入对称归一化（symmetric normalization），利用![](https://cdn.nlark.com/yuque/__latex/bc1f966501b658800daef8b40f03603f.svg#card=math&code=%5Ctilde%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Ctilde%7BA%7D%5Ctilde%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D&height=24&width=74)代替。这样，我们就推导出了GCN的逐层传播的公式！<br />![](https://cdn.nlark.com/yuque/__latex/23c660b321bd73a6cbed5833af91c954.svg#card=math&code=H%5E%7B%28l%2B1%29%7D%20%3D%20%5Csigma%5Cleft%28%5Ctilde%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Ctilde%7BA%7D%5Ctilde%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7DH%5E%7B%28l%29%7DW%5E%7B%28l%29%7D%20%5Cright%29.%20%20%20%20%20%20%5Ctag%7BKipf%20and%20Welling.%202017.%20%282%29%7D&height=39&width=643)

现在我们再考察单个节点，来理解一下对称归一化：<br />![](https://cdn.nlark.com/yuque/__latex/3771c6c876cd1b5da038834a0fdb7d92.svg#card=math&code=%5Ctilde%7Bh%7D_u%5E%7B%28l%2B1%29%7D%20%3D%20%5Csum_%7Bv%5Cin%20N%28u%29%20%5Ccup%20%5C%7Bv%5C%7D%7D%20%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Ctilde%7Bd%7D_u%20%5Ccdot%20%5Ctilde%7Bd%7D_v%7D%7D%20h_v%5E%7B%28l%29%7D.%20%5Ctag%7B4%7D&height=50&width=643)<br />其实就是从除以![](https://cdn.nlark.com/yuque/__latex/dab8facd94150ed286eafcd27fea84a7.svg#card=math&code=%5Ctilde%7Bd%7D_u&height=19&width=17)变成了除以![](https://cdn.nlark.com/yuque/__latex/dab8facd94150ed286eafcd27fea84a7.svg#card=math&code=%5Ctilde%7Bd%7D_u&height=19&width=17)和![](https://cdn.nlark.com/yuque/__latex/7cbb4c82bff1ec745001cefa61b4a394.svg#card=math&code=%5Ctilde%7Bd%7D_v&height=19&width=16)的几何平均。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/384931/1569748388139-58d3ed72-cd6c-46f4-9eea-00d7385466e3.png#align=left&display=inline&height=442&name=image.png&originHeight=442&originWidth=1038&search=&size=49821&status=done&width=1038)

可以看这个例子来对比一下三种的区别，注意第二和第三个图中加入了Self-connections。图中的箭头只是表明aggregation，并不代表是有向边。


<a name="T5go5"></a>
# 为什么要对称归一化

上面我们讲了因为数量级会爆炸，所以要求均值而不是求和，又因为求均值归一化不够漂亮，所以要对称归一化。现在我们直观地来理解一下对称归一化。

![image.png](https://cdn.nlark.com/yuque/0/2019/png/384931/1569756049034-a04f0d8f-3129-4a64-8801-ba91a58428f8.png#align=left&display=inline&height=290&name=image.png&originHeight=407&originWidth=994&search=&size=96162&status=done&width=709)

我们的中心节点![](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg#card=math&code=u&height=10&width=8)要从一众邻居节点接受信息，如果这个节点有很多邻居，那么他接受到的信息量就很大、很杂，因此，每一个邻居传过来的信息的重要性就会变低。

现在我们换个视角，从信息的发出方![](https://cdn.nlark.com/yuque/__latex/9e3669d19b675bd57058fd4664205d2a.svg#card=math&code=v&height=10&width=7)来看。节点![](https://cdn.nlark.com/yuque/__latex/9e3669d19b675bd57058fd4664205d2a.svg#card=math&code=v&height=10&width=7)要把信息传出给它的所有邻居节点，如果它的邻居众多，那它可能就是个发小广告的，因此它传出去的信息的重要性就会变低。

这样，我们看从![](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg#card=math&code=u&height=10&width=8)到![](https://cdn.nlark.com/yuque/__latex/9e3669d19b675bd57058fd4664205d2a.svg#card=math&code=v&height=10&width=7)的这一条信息，其的重要性就要打一定的折扣，![](https://cdn.nlark.com/yuque/__latex/7b774effe4a349c6dd82ad4f4f21d34c.svg#card=math&code=u&height=10&width=8)觉得这条信息的重要性是原来的![](https://cdn.nlark.com/yuque/__latex/334ffe6aeaccd3ef21a79b5ae1b924e1.svg#card=math&code=%5Cfrac%7B1%7D%7B%5Ctilde%7Bd%7D_u%7D&height=39&width=23)，而![](https://cdn.nlark.com/yuque/__latex/9e3669d19b675bd57058fd4664205d2a.svg#card=math&code=v&height=10&width=7)觉得是![](https://cdn.nlark.com/yuque/__latex/740e82392883b952cbfdd134121d8979.svg#card=math&code=%5Cfrac%7B1%7D%7B%5Ctilde%7Bd%7D_v%7D&height=39&width=22)。为了不打架，我们就取个平均![](https://cdn.nlark.com/yuque/__latex/63c2db0536f79e3b6da3a5ce863eca9e.svg#card=math&code=%5Cfrac%7B1%7D%7B%5Csqrt%7B%5Ctilde%7Bd%7D_u%20%5Ccdot%20%5Ctilde%7Bd%7D_v%7D%7D&height=51&width=65)好了😄。

<a name="DwVf0"></a>
# 后记

其实个人觉得，Kipf的博客中介绍的GCN比论文要好太多了，论文强行从图谱理论进行解释，充斥着奇怪的假设与近似，可以参考知乎上的这个[讨论](https://zhuanlan.zhihu.com/p/60014316)。但奈何GCN又简洁效果又好，火也是有道理的，而且还推动了整个图神经网络的发展。

这篇文章根据Kipf的博客，从直觉出发对GCN进行了推导。之后有时间，再来为大家介绍一下我了解的图谱理论的一些皮毛。主要想介绍一下热传导方程与图卷积网络的相似之处，真的很有意思。比如热传导方程有Laplace算子，求解要用傅立叶变换；而图卷积里也有Laplace矩阵，推导也涉及到傅立叶变换，这其中究竟隐藏了怎样的秘密？且听下回分解🐦。

{% pdf /pdf/Basic-GCN.pdf %}