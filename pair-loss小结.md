---
typora-root-url: ./-lo
---

# 								pairwise loss

#### 1 max_margin_loss(hinge loss)

**二分类下的基本形式**：
$$
L(\hat{y})=    max(0,1-y*\hat{y})
$$
一般用于二分类。如果真实标签$y$取值是1的话，公式退化为
$$
L(\hat{y})=    max(0,1-\hat{y})    \\
          =\left\{
 \begin{gathered}
  1- \hat{y}       & \hat{y}<1,        \\
  0       & \hat{y}>=1        \\

 \end{gathered}
\right.
$$
![img](https://pic3.zhimg.com/80/v2-3c6aa9626ee8e4609b0d7c5712baf624_720w.jpg?source=1940ef5c)

即把1当作一个阈值。如果预测值超过这个阈值，认为该样本已经足够好，就不需要再去学习了。而是专注于学习那些预测值还没有达到该阈值的样本。尽量让预测值去逼近这个阈值，学习该预测值和阈值之间的差值。因为该损失像一本打开的书，所以称为合页损失函数。

负样本场景同理 。都是只要达到这个阈值了就不去再学习。只有预测值不达标的样本才会产生损失。本质都是让预测值从一侧接近我们设定的阈值。

**更通用的形式**
$$
L(\hat{y})=    max(0,1-z)    \\
$$
本质上，hinge loss学习的是一个分数与我们指定的阈值之间的差距。如果已经超过了我们设定的这个"足够好"的标准，就不需要再去学习了。否则需要去学习不满足的那部分：阈值 -当前分数。 

 优点1： 相比0/1损失, 会对未满足阈值的样本进一步学习。

![See the source image](https://www.researchgate.net/profile/Yoonkyung_Lee/publication/45283473/figure/download/fig1/AS:340863369662470@1458279568961/The-solid-line-is-the-0-1-loss-and-the-dashed-line-is-the-hinge-loss-in-terms-of-the.png)



缺点1：只会对未满足阈值的样本学习。对满足阈值或者超过阈值的样本，没有足够的限制。因此需要很好的定义分数和阈值。

可以通过灵活调整阈值和分数z的定义来满足不同的使用场景。

ex: 认为分数大于0就足够好，可以把阈值设置成0。

ex: 二分类场景标签取+1/-1的场景，可以定义分数是$z=y*\hat{y}$, 反应预测值和真值的相似程度。

ex:二分类标签取1/0的场景，可以定义阈值z是0。当y=1时，定义预测值大于y就足够好了。y=0时，定义预测值小于0就足够好了。即分别定义了2种条件下的z。可统一写做：
$$
z=      \left\{ \begin{gathered} \hat{y}-y       & y=1,        \\  y-\hat{y}       & y=0        \\ \end{gathered}\right. =sign(y)*(\hat{y}-y)\\  
$$
按阈值取后，损失函数如图：$L(\hat{y})=    max(0,0-z)    = max(0, -z)$。相比标签为+-1的场景，预测值的阈值变成了0。

![img](https://img-blog.csdnimg.cn/202005182200369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3JpY2hhcmRfY2hl,size_16,color_FFFFFF,t_70#pic_center)



**pair-wise形式**

在推荐等rank场景下，常常更关注2个样本的相对大小而非绝对大小。因此在仍然用hinge loss作为损失函数的情况下，一般用正负样本之间的差距作为我们学习的目标，希望正样本的分值比负样本的分值高。因此定义分值$z=s_+-s_-$。对于选定的阈值margin, hinge loss可以写做：
$$
L(\hat{y})=    max(0,margin-z)    \\  =    max(0,margin- (s_+-s_-)) \\  =    max(0,margin +s_- - s_+))
$$
当正样本的分值比负样本的分值，超过设定的阈值margin后，模型会认为该pair已经能够较好的区分正负样本了。因此不再对这个pair进行学习了。 因为该损失函数是在最大化正负样本的分数间隔（直到满足阈值为止），因此也被叫做max margin loss。在pair-wise的学习中比较常见。较著名的比如TransE、Pinsage等算法都采用了该损失函数，可以使正负样本有较好的区分度。



to be contionued:  另一个很类似的算法是BRP loss. 我们下期继续分享。

