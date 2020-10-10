[TOC]

# Google多任务模型

## MMoE

### MTL背景

> MTL(Multi-Task Learning)有很多形式：联合学习(joint learning)、自主学习(learning to learn)和带有辅助任务的学习(learning with auxiliary task，如ESMM系列)等等。一般来说，优化多个损失函数就等同于进行多任务学习(与单任务学习相反)。

> MLT 主要有两种形式，一种是基于参数的共享，另一种是基于约束的共享。
>
> - hard参数共享
>
>   参数共享的形式在基于神经网络的 MLT 中非常常见，其在所有任务中共享隐藏层并同时保留几个特定任务的输出层。
>
>   这种方式有助于降低过拟合风险，因为同时学习的任务越多，模型找到一个含有所有任务的表征就越困难，从而过拟合某个特定任务的可能性就越小，ESMM 就属于这种类型的 MLT。
>
> - soft参数共享
>
>   每个任务都有自己的参数和模型，最后通过对不同任务的参数之间的差异施加约束。比如可以使用L2进行正则、迹范数（trace norm）等。

> 为什么 MLT 有效呢？主要有以下几点原因：
>
> 1. 多任务一起学习时，会互相增加噪声，从而提高模型的泛化能力；
> 2. 多任务相关作用，逃离局部最优解；
> 3. 多任务共同作用模型的更新，增加错误反馈；
> 4. 降低了过拟合的风险；
> 5. 类似 ESMM，解决了样本偏差和数据稀疏问题，也可以用来解决冷启动问题。

### MTL适用条件

> MTL 的目标在于**通过利用包含在相关任务训练信号中特定领域的信息来提高泛化能力**。

> 什么是相关任务呢？有以下几个不严谨的解释：
>
> 1. 使用相同特征做判断的任务；
> 2. 任务的分类边界接近；
> 3. 预测同个个体属性的不同方面比预测不同个体属性的不同方面更相关；
> 4. 共同训练时能够提供帮助并不一定相关，因为加入噪声有时也可以增加泛化能力。

> 作者在论文中给出了多任务模型在相关性不同的数据集上的表现，证明：相关性越低，多任务学习的效果越差，如图：

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/1.jpg)

> 在实际过程中，如何去识别不同任务之间的相关性也是非常难的。基于此，作者提出了 MMoE 框架，旨在构建一个兼容性更强的多任务学习框架。

### 模型

#### Shared-Bottom model

> ESMM 模型就是基于 shared-bottom 的多任务模型，这篇文章把该框架作为多任务模型的 baseline，其结构如下图所示：
>
> 所有任务共享底层网络，并同时保留几个特定任务的输出层。

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/2.jpg)

#### One-gate MoE Layer

> One-gate MoE layer 是将隐藏层划分为多个专家(expert)子网，同时接入一个 Gate 网络，将各个子网的输出和输入信息进行组合，并将得到的结果进行相加。

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/3.jpg)

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/5.jpg)

#### Multi-gate MoE model

> One-gate MoE 能够实现不同数据多样化使用共享层，但针对不同任务而言，其使用的共享层是一致的。这种情况下，如果任务相关性较低，则会导致模型性能下降。所以，作者在此基础上提出了 MMoE 模型，为每个任务都设置了一个 Gate 网路，旨在使得不同任务和不同数据可以多样化的使用共享层，其模型结构如下：

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/4.jpg)

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/6.jpg)

> 这种情况下，每个 Gate 网络都可以根据不同任务来选择专家网络的子集，所以即使两个任务并不是十分相关，那么经过 Gate 后也可以得到不同的权重系数。此时，MMoE 可以充分利用部分 expert 网络的信息，近似于单个任务；而如果两个任务相关性高，那么 Gate 的权重分布相差会不大，会类似于一般的多任务学习。

### 实验结果

#### MTL模型在不同相关性任务下的loss分布

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/7.jpg)

> 论文比较了不同 MLT 模型在不同相关性任务下的loss分布，其可以反应模型的鲁棒性，从图中可以看出：
>
> 1. shared-bottom模型的表现方差高于OMoE和MMoE；
> 2. 虽然在correlation=1的数据集下，OMoE和MMoE的表现接近，但是在correlation=0.5的数据集下，OMoE的表现有明显的下降，这也证明了MMoE的multi-gate structure能够有效的解决任务差异冲突带来的局部最小值；

#### 数据集的效果

> 论文比较了各个模型在两个数据集上的效果，如下：

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/8.jpg)

#### Google推荐系统效果

> 论文比较了各个模型在Google推荐系统上的效果，如下：

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/9.jpg)

> 此外，论文展示了gate在两个任务上的分布，satisfaction任务的label比engagement任务的label更加稀疏，所以satisfaction的gate分布更加偏向于单个expert。

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/10.jpg)

### 代码

https://github.com/drawbridge/keras-mmoe

## MMoE在YouTube的应用

参考：https://zhuanlan.zhihu.com/p/126172558



## MoSE

### 背景

> MMoE多任务学习框架由于其模型的简单，已经在工业界进行了大量的应用。然而，因为MMoE中experts的足够“简单”，使其在表达各个任务的同时，不能很好地对用户的行为序列进行解析，因此就会导致MMoE模型在进行多任务学习时，不能很好地对序列行为数据进行有效表达。
>
> 基于此，Google研究者提出了MoSE模型，在expert中集成了Long Short-Term Memory (LSTM)，旨在充分利用用户的序列行为数据，更好的达到业务需求。

### 模型

> MoSE模型结构如下：
>
> 1. 模型采用LSTM做share bottom，充分利用user的行为序列数据；
> 2. 顺序expert层的混合，通过使用LSTM代替全连接网络，进一步增加了MoE层，更好地处理序列数据。
> 3. 用门控网络对expert的输出进行门控。每个门控网络可以学习“选择”一个子集的expert使用条件输入的例子，这允许在不同的变量之间建模复杂的交互作用。
> 4. 每个任务的Tower网络采用LSTM机制；

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/11.jpg)

### 实验结果

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/12.jpg)

> 论文比较了MoSE和七个备选模型在G Suite数据集上的效果，结论如下：
>
> 1. MoSE的模型性能优于其他模型，尤其是在复杂的真实数据集上。
> 2. 序列模型的性能优于非序列模型，这说明了在用户活动流中显式地建模序列依赖关系的必要性。
> 3. MoSE性能优于其他序列模型。这说明MoSE模块能够有效地处理用户活动流中的稀疏变量和异构数据源之间复杂的交互等问题。
> 4. 与其他非序列模型相比，MMoE本身并没有显示出显著的优势。当在MoSE中使用顺序expert时，expert混合框架是最有益的，因为用户活动流中的大多数复杂性似乎源于顺序复杂性和稀疏性。

### 缺陷

> MoSE模型在序列化建模中可以取得不错的效果，但是采用LSTM建模，模型的运行时间长、速度慢，线上响应也会是个问题。
