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

## PLE-Tencent

### 背景

> 当不同学习任务之间较为相关时，多任务学习可以通过任务之间的信息共享提升学习的效率。但通常情况下，任务之间的相关性并不强，有时候甚至是冲突的，此时应用多任务学习可能带来负迁移(negative transfer)现象，会影响网络的表现。
>
> 此前已经有部分研究来减轻负迁移现象，如谷歌提出的MMoE模型。但通过实验发现，多任务学习中往往还存在seesaw phenomenon，即：多任务学习相对于多个单任务学习的模型，往往能够提升一部分任务的效果，同时牺牲另外部分任务的效果。即使通过MMoE这种方式减轻负迁移现象，seesaw phenomenon仍然是广泛存在的。
>
> 论文提出了Progressive Layered Extraction (简称PLE)，来解决多任务学习的seesaw phenomenon。

### 多任务学习模型

> 论文中将MTL模型分为了Single-Level MTL Models和Multi-Level MTL Models，如下：


![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/13.jpg)

#### Single-Level MTL Models

- a. Hard Parameter Sharing

  这是最为常见的MTL模型，不同的任务底层的模块是共享的，然后共享层的输出分别输入到不同任务的独有模块中，得到各自的输出。当两个任务相关性较高时，用这种结构往往可以取得不错的效果，但任务相关性不高时，会存在负迁移现象，导致效果不理想。

- b. Asymmetry Sharing

  不同任务的底层模块有各自对应的输出，但其中部分任务的输出会被其他任务所使用，而部分任务则使用自己独有的输出。哪部分任务使用其他任务的输出，则需要人为指定。

- c. Customized Sharing

  不同任务的底层模块不仅有各自独立的输出，还有共享的输出。

- d. MMoE

  上面介绍过

- e. CGC

  论文提出的结构，下面将详细介绍。

#### Multi-Level MTL Models

- f. Cross-Stitch Network

  具体参考：

- g. Sluice Network

  具体参考：

- h. ML-MMoE

  MMoE模型的多级叠加

- i. PLE

  CGC模型的多级叠加

### PROGRESSIVE LAYERED EXTRACTION

#### seesaw phenomenon

> 论文主要基于腾讯视频推荐中的多任务学习为例进行介绍，其视频推荐架构如下图：


![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/14.jpg)

> 这里主要关注VCR和VTR两个任务，VCR任务是视频播放完成度，即视频播放时长/视频总时长，这个是回归问题，并以MSE作为评估指标。VTR表示此次观看是否是一次有效观看，即观看时长是否在给定的阈值之上，这是二分类问题（如果没有观看，样本Label为0），并以AUC为评估指标。
>
> 两个任务之间的关系比较复杂：首先，VTR的标签是播放动作和VCR的耦合结果，因为只有观看时间超过阈值的播放动作才被视为有效观看；其次，播放动作的分布更加复杂，在存在WIFI时，部分场景有自动播放机制，这些样本就有较高的平均播放概率，而没有自动播放且需要人为显式点击的场景下，视频的平均播放概率则较低。
>
> 论文对比了上述所有结构的MTL在腾讯视频VCR和VTR两个任务上相对单任务模型的离线训练结果：


![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/15.jpg)

> 可以看到，几乎所有的网络结构都是在一个任务上表现优于单任务模型，而在另一个任务上表现差于单任务模型。尽管MMoE有了一定的改进，在VTR上取得了不错的收益，但在VCR上的收益接近于0。
>
> MMoE模型存在以下两方面的缺点，首先，MMoE中所有的Expert是被所有任务所共享的，这可能无法捕捉到任务之间更复杂的关系，从而给部分任务带来一定的噪声；其次，不同的Expert之间也没有交互，联合优化的效果有所折扣。

#### Customized Gate Control(CGC)


![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/16.jpg)

> CGC网络结构是Customized Sharing和MMoE的结合版本，每个任务有共享的Expert和独有的Expert。每个Expert有多个sub-network即experts，其数量可以作为参数调节。
>
> 以任务A来说，将Experts A里面的多个Expert的输出以及Experts Shared里面的多个Expert的输出，通过类似于MMoE的门控机制之后输入到任务A的上层网络中，计算公式如下：


![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/17.jpg)

#### Progressive Layered Extraction(PLE)


![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/18.jpg)

> PLE是在CGC基础上考虑了不同Expert的交互，可以看作是Customized Sharing和ML-MMOE的结合版本。
>
> 下层模块中增加了多层Extraction Network。在每一层Extraction Network，共享Experts不断吸收各自独有的Experts之间的信息，而任务独有的Experts则从共享Experts中吸收有用的信息，具体计算和CGC一样。


![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/19.jpg)

### MTL loss优化

> 传统的MTL的损失是各任务损失的加权和，如下：

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/20.jpg)

> 而在腾讯视频场景下，不同任务的样本空间是不一样的，比如计算视频的完成度，必须有视频点击行为才可以。不同任务的样本空间如下图所示：

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/21.jpg)

> 本文是在Loss上进行优化，不同的任务仍使用其各自样本空间中的样本，如下：

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/29.jpg)

### 实验结果

#### 离线训练结果

##### VCR/VTR

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/22.jpg)

> VCR 和 VTR之间的关系复杂，从图中可以看出：
>
> CGC和PLE的效果均显著优于其他模型，PLE效果最好；
>
> 许多模型存在seesaw phenomenon，VCR提升但VTR下降，或者，VTR提升但VCR下降；
>
> MMoE均能提升VTR和VCR，但效果不显著；

##### CTR/VCR

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/23.jpg)

> CTR和VCR之间是正相关关系，关系简单，从图中可以看出：
>
> 除了Cross-Stitch Network，其他模型均在CTR和VCR任务上取得正向效果，没有出现seesaw phenomenon；
>
> CGC和PLE的效果均显著优于其他模型，PLE效果最好；

#### 线上ABtest效果

> 论文在线上进行了4周的ABtest实验，主要优化目标是VCR和VTR，结果如下：
>
> MTL模型比单任务模型效果更好；
>
> CGC和PLE的效果均显著优于其他模型，PLE效果最好；

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/24.jpg)

#### 更多任务表现结果

> 除了VCR和VTR，作者还引入了SHR和CMR两个任务到MTL框架中，比较了CGC和PLE的效果，从图中可以看出：
>
> CGC和PLE在多任务学习中的效果均显著优于单任务模型；
>
> CGC和PLE在超过两个子任务的多任务学习中，可以有效地避免seesaw phenomenon和负迁移；
>
> PLE的效果优于CGC；

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/25.jpg)

#### 公开数据集表现

> 作者在三个公开数据集上比较了Hard Parameter Sharing、MMoE和PLE的效果，从图中可以看出：
>
> Hard Parameter Sharing和MMoE均存在seesaw phenomenon，而PLE则表现很好，有效地消除了seesaw phenomenon；

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/26.jpg)

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/27.jpg)

#### Expert utilization analysis

> 为了公平对比，作者采用了single-level的PLE和ML-MMoE，然后可视化了CGC、MMoE、PLE和ML-MMoE的expert的utilization，如图所示：

![image](https://github.com/ShaoQiBNU/Google_MTL/blob/main/img/28.jpg)

> 从图中可以看出：
>
> CGC里VTR和VCR的expert权重有着显著不同，而MMoE中几乎相似，这也表明CGC效果优于MMoE；
>
> MMoE和ML-MMoE所有的expert权重几乎不为0，这也表明：没有先验知识的情况下，MMOE and ML-MMOE很难收敛到CGC和PLE的结构，即便理论上存在可能性；
>
> 与CGC相比，PLE的shared experts对Tower有更大的影响，尤其是在VTR任务中。PLE性能优于CGC，这表明共享更高级的更深层表示的价值。换句话说，为了在任务之间共享某些更深的语义表示，PLE提供了更好的联合路由和学习方案。


## 其他多任务模型

https://zhuanlan.zhihu.com/p/268359893
