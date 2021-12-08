#### 推导FM O(k,n)公式

假设单个样本共n个特征，其对应的特征值向量是
$$
\vec{x}=[x_1,x_2,...x_i,...,x_n]
$$
其中$x_i$代表特征$i$的取值。这n个特征，每个特征都对应一个K维的向量$\vec{v_i}$，相当于特征$i$对应的embedding。用来和其他特征对应的向量做两两特征交叉(通过内积形式)：
$$
\vec{v_i}=[v_{i1},v_{i2},...v_{ik},...v_{iK}]
$$
因此这n个特征会维护一个[n,k]的特征向量矩阵，每一行代表特征$i$对应的一个k维向量。而特征矩阵的每k列则对应一个（n，1）的列向量。每个元素对应每个特征的第k个维度。
$$
V（n,k）= \left\{
 \begin{matrix}
 \cdots      & \vec{v_1}        & \cdots        \\
 \cdots      & \vec{v_2}        & \cdots        \\
 \vdots      & \vdots           & \vdots    \\
 \cdots     &  \vec{v_i}       & \cdots    \\
  \vdots      & \vdots           & \vdots    \\
 \cdots      & \vec{v_n} & \cdots 
 \end{matrix}
 \right\}= \left\{
 \begin{matrix}
 \cdots      & \cdots & v_{1k}        & \cdots    & \cdots      \\
 \cdots      & \cdots & v_{2k}        & \cdots    & \cdots    \\
 \vdots      & \vdots & \vdots        & \vdots   & \vdots    \\
 v_{i1}      & \cdots &  v_{ik}       & \cdots   &  v_{iK}  \\
  \vdots      & \vdots  & \vdots          & \vdots & \vdots    \\
 \cdots      & \cdots& v_{nk} & \cdots & \cdots
 \end{matrix} 
 \right\} \\ \qquad \qquad \qquad  \qquad   \quad     \qquad \qquad \qquad 第k列：(n,1)
$$
对有n个特征，每个特征维护一个k维向量的FM来说，二阶特征交叉的得分$y_{FM-order2}$，是这n个特征对应的向量两两交叉的结果：
$$
y_{FM-order2}= \sum_{i}^{n}\sum_{j!=i}^{n}x_ix_j<\vec{v_i},\vec{v_j}>
$$
直接计算是$O(kn^2)$，(n个特征两两交互$O(n^2)$次,每次交互，算一次$k$维向量的内积)。但是可以简化：
$$
y_{FM-order2}= \sum_{i}^{n}\sum_{j!=i}^{n}x_ix_j<\vec{v_i},\vec{v_j}> \\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j<\vec{v_i},\vec{v_j}> - \sum_{i}^{n}\sum_{i}^{n}x_ix_i<\vec{v_i},\vec{v_i}>)
$$
第一项相当于改成把相同项的交互$x_ix_i$，以及重复的$x_ix_j$, $x_jx_i$都算上。原来是$x_1x_2$，$x_1x_3$，$x_2x_3$之间的交互，现在的第一项改成$x_1x_1$，$x_1x_2$，$x_1x_3$，$x_2x_1$，$x_2x_2$，$x_2x_3$，$x_3x_1$，$x_3x_2$，$x_3x_3$。两两交互都算了2遍。所以需要减去自交互项后，再除以2。

进一步，把两两特征向量内积$<\vec{v_i},\vec{v_j}>$写成求和形式：$\sum_{k} v_{ik}v_{jk}$：
$$
y_{FM-order2}= \sum_{i}^{n}\sum_{j!=i}^{n}x_ix_j<\vec{v_i},\vec{v_j}> \\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j<\vec{v_i},\vec{v_j}> - \sum_{i}^{n}\sum_{i}^{n}x_ix_i<\vec{v_i},\vec{v_i}>)\\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik},v_{jk} - \sum_{i}^{n}\sum_{i}^{n}x_ix_i\sum_{k}v_{ik},v_{ik})\\
=   \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik},v_{jk} - \sum_{i}^{n}x_i^2\sum_{k}v_{ik}^2)\\
$$
把沿着k的sum提出来，假设每一项k值固定，先只算ij维上的sum:
$$
y_{FM-order2}= \sum_{i}^{n}\sum_{j!=i}^{n}x_ix_j<\vec{v_i},\vec{v_j}> \\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j<\vec{v_i},\vec{v_j}> - \sum_{i}^{n}\sum_{i}^{n}x_ix_i<\vec{v_i},\vec{v_i}>)\\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik}v_{jk} - \sum_{i}^{n}\sum_{i}^{n}x_ix_i\sum_{k}v_{ik},v_{ik})\\
=   \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik}v_{jk} - \sum_{i}^{n}x_i^2\sum_{k}v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( \sum_{i}^{n}\sum_{j}^{n}x_ix_jv_{ik}v_{jk} - \sum_{i}^{n}x_i^2v_{ik}^2)\\
$$
再把第一项的i,j的加和分别提出来:
$$
y_{FM-order2}= \sum_{i}^{n}\sum_{j!=i}^{n}x_ix_j<\vec{v_i},\vec{v_j}> \\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j<\vec{v_i},\vec{v_j}> - \sum_{i}^{n}\sum_{i}^{n}x_ix_i<\vec{v_i},\vec{v_i}>)\\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik}v_{jk} - \sum_{i}^{n}\sum_{i}^{n}x_ix_i\sum_{k}v_{ik},v_{ik})\\
=   \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik}v_{jk} - \sum_{i}^{n}x_i^2\sum_{k}v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( \sum_{i}^{n}\sum_{j}^{n}x_ix_jv_{ik}v_{jk} - \sum_{i}^{n}x_i^2v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( \sum_{i}^{n}x_iv_{ik}\sum_{j}^{n}x_jv_{jk} - \sum_{i}^{n}x_i^2v_{ik}^2)\\
$$
其中$\sum_{i}^{n}x_iv_{ik}$和$\sum_{j}^{n}x_jv_{jk}$，值完全相同，$i$都从1取到$n$。本质上相当于样本$\vec{x}$和特征矩阵的第$k$列对应的列向量的内积
$$
\sum_{i}^{n}x_iv_{ik}= x_1v_{1k} + x_2v_{2k} + \cdots +x_nv_{nk} = < \vec{x},\vec{V[k]}>
$$
其中列向量$\vec{V[k]}$对应特征矩阵的第$k$列：
$$
V（n,k）= \left\{
 \begin{matrix}
 \cdots      & \vec{v_1}        & \cdots        \\
 \cdots      & \vec{v_2}        & \cdots        \\
 \vdots      & \vdots           & \vdots    \\
 \cdots     &  \vec{v_i}       & \cdots    \\
  \vdots      & \vdots           & \vdots    \\
 \cdots      & \vec{v_n} & \cdots 
 \end{matrix}
 \right\}= \left\{
 \begin{matrix}
 \cdots      & \cdots & v_{1k}        & \cdots    & \cdots      \\
 \cdots      & \cdots & v_{2k}        & \cdots    & \cdots    \\
 \vdots      & \vdots & \vdots        & \vdots   & \vdots    \\
 v_{i1}      & \cdots &  v_{ik}       & \cdots   &  v_{iK}  \\
  \vdots      & \vdots  & \vdots          & \vdots & \vdots    \\
 \cdots      & \cdots& v_{nk} & \cdots & \cdots
 \end{matrix} 
 \right\} \\ \qquad \qquad \qquad  \qquad   \quad     \qquad \qquad \qquad V[k]：(n,1)
$$
而第二项$\sum_{i}^{n}x_i^2v_{ik}^2$可以看做是特征值向量$\vec{x}$和特征矩阵列向量$\vec{V[k]}$分别平方后的内积：
$$
\sum_{i}^{n}x_i^2v_{ik}^2= x_1^2v_{1k}^2 + x_2^2v_{2k}^2 + \cdots +x_n^2v_{nk}^2 = < \vec{x}^2,\vec{V[k]}^2>
$$
所以这n个特征两两交叉后，得到的2阶预测分数，最终可以简化为：
$$
y_{FM-order2}= \sum_{i}^{n}\sum_{j!=i}^{n}x_ix_j<\vec{v_i},\vec{v_j}> \\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j<\vec{v_i},\vec{v_j}> - \sum_{i}^{n}\sum_{i}^{n}x_ix_i<\vec{v_i},\vec{v_i}>)\\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik}v_{jk} - \sum_{i}^{n}\sum_{i}^{n}x_ix_i\sum_{k}v_{ik},v_{ik})\\
=   \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik}v_{jk} - \sum_{i}^{n}x_i^2\sum_{k}v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( \sum_{i}^{n}\sum_{j}^{n}x_ix_jv_{ik}v_{jk} - \sum_{i}^{n}x_i^2v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( \sum_{i}^{n}x_iv_{ik}\sum_{j}^{n}x_jv_{jk} - \sum_{i}^{n}x_i^2v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( < \vec{x},\vec{V[k]}>^2 - < \vec{x}^2,\vec{V[k]}^2>)\\
$$
而特征$\vec{x}$和特征矩阵$V$进行矩阵相乘，得到的第k个位置的元素，就是x和$V$的列向量$\vec{V[k]}$的内积：
$$
\vec{x}_{(1n)}*\vec{V}_{(nk)}=  [c_1,c_2,\cdots ,c_k]_{(1,k)} \\
= [< \vec{x},\vec{V[1]}>,< \vec{x},\vec{V[2]}>,\cdots ,< \vec{x},\vec{V[k]}>]
$$
所以第一项是$\vec{x}_{(1n)}*\vec{V}_{(nk)}$矩阵相乘后的所有值相加，之后再平方：
$$
\sum_{k}( < \vec{x},\vec{V[k]}>^2)\\
= (sum( \vec{x}_{(1n)}*\vec{V}_{(nk)})) ^2
$$
而第二项类似。只不过x和V在矩阵相乘前，所有元素先平方：
$$
\vec{x^2}_{(1n)}*\vec{V}^2_{(nk)}=  [d_1,d_2,\cdots ,d_k]_{(1,k)} \\
= [ < \vec{x}^2,\vec{V[1]}^2>, < \vec{x}^2,\vec{V[2]}^2>,\cdots , < \vec{x}^2,\vec{V[k]}^2>]
$$
所以最终沿维度k进行相加，每一项都相当于矩阵乘完的元素相加。最后再计算差。得到的是FM的这n个特征，二阶特征两两交叉后的总score：
$$
y_{FM-order2}= \sum_{i}^{n}\sum_{j!=i}^{n}x_ix_j<\vec{v_i},\vec{v_j}> \\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j<\vec{v_i},\vec{v_j}> - \sum_{i}^{n}\sum_{i}^{n}x_ix_i<\vec{v_i},\vec{v_i}>)\\
=  \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik}v_{jk} - \sum_{i}^{n}\sum_{i}^{n}x_ix_i\sum_{k}v_{ik},v_{ik})\\
=   \frac{1}{2}( \sum_{i}^{n}\sum_{j}^{n}x_ix_j \sum_{k}v_{ik}v_{jk} - \sum_{i}^{n}x_i^2\sum_{k}v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( \sum_{i}^{n}\sum_{j}^{n}x_ix_jv_{ik}v_{jk} - \sum_{i}^{n}x_i^2v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( \sum_{i}^{n}x_iv_{ik}\sum_{j}^{n}x_jv_{jk} - \sum_{i}^{n}x_i^2v_{ik}^2)\\
=   \frac{1}{2}  \sum_{k}( < \vec{x},\vec{V[k]}>^2 - < \vec{x}^2,\vec{V[k]}^2>)\\
=   \frac{1}{2}  (  (sum( \vec{x}*V))^2 - sum( \vec{x^2}*V^2) )
$$
最终的复杂度是每个维度的向量内积<x,V[k]>的$O(n)$，以及最后的k维加和：$O(kn)$

#### DeepFM里的实现

对DeepFM来说，每个原始特征对应一个域(Field):每个离散特征对应的域，包含该特征的所有取值。连续特征对应的域只有它自身。在进行二阶交互时，每个连续特征映射为一个embedding，每个离散特征维护对应的C个embedding向量（C:该连续特征的取值）。算二阶交互时，每个样本的每个特征域先映射到对应向量，之后不同域向量之间22内积交互，得到最终的二阶交互得分。如果样本的原始特征域是[x1,x2,x3], 对应的embedding是[$\vec{e1}$,$\vec{e2}$,$\vec{e3}$]。每个域的embedding就相当于该特征对应的特征向量v，去和其他域的特征向量通过内积交互加和
$$
y_{FM-order2}= \sum_{i}^{n}\sum_{j!=i}^{n}x_ix_j<\vec{e_i},\vec{e_j}> \\
$$
对离散特征，根据取值取到对应的embedding，对应的xi是1。得到每个特征的embedding后，先乘上对应取值$x_i$，就可以算域向量的对应位置的交互了(为方便先忽略$x_i$)：$\sum_{i}^{n}\sum_{j!=i}^{n}<\vec{e_i},\vec{e_j}>$。

对向量$e$=[$\vec{e1}$,$\vec{e2}$,$\vec{e3}$]=[ $v_1 ，v_2， v_3$,| $v_4 ，v_5， v_6$ | $v_7 ，v_8， v_9$]，该值本质上是每个embedding向量第f个维度的所有元素22相乘后得到的值求和，最后在所有位置上求和：$[v_1*v_4+v_2*v_5+..+v_1*v_7+v_2*v_8...+v_4*v_7]$ , 如果单独看每个维度，是V的该列元素22交互的和，再相加：$[v_1*v_4+v_1*v_7+..] + $$[v_2*v_5+v_2*v_8..]$ +...。每个位置对应的值是所有域的embedding在该维度的22交互（V向量的列元素22交互）。f=1的位置对应的值是：$[v_1*v_4+v_1*v_7+..]$，f=2的位置对应的值：$[v_2*v_5+v_2*v_8+..$]。

对于每个固定位置，可以通过square_of_sum - sum_of_square得到每个f处的交互值。最后再沿着f维相加，就可以得到最终的score。square_of_sum是所有向量的相加得到的向量：$\vec{e_{sum}}=\sum_{i}\vec{e_i}$，每个位置再平方。sum_of_square是所有embedding先元素平方，再把平方后的各向量相加：$\sum_{i}\vec{e_i}^2 =[\vec{e_1}^2 + \vec{e_2}^2,...+\vec{e_n}^2]$。相减可得各域的embedding在每个维度f上的22元素交互的结果(1,k)：
$$
\sum_{i}^{n}\sum_{j!=i}^{n}<\vec{e_i},\vec{e_j}>  （每个维度f）\\
  =\frac{1}{2}((\sum_{i}\vec{e_i})^2 - \sum_{i}\vec{e_i}^2 )    （每个维度f）\\
  = [(v_1+v_4+...)^2,(v_2+v_5+...)^2 ,...]_{1,k} - [(v_1^2+v_4^2+..), (v_2^2+v_5^2+..),...]_{1,k}\\
  = [(v_1*v_4+v_1*v_7+...),(v_2*v_5+v_2*v_8+...), .]_{1,k} \qquad(每个embedding的维度f对应一个值：所有原始特征在该维度上的bi-interaction)
$$
可以直接保留这k个位置的所有特征交互值(1,k)，作为新的特征拼接其他特征。每个新的交叉特征，是所有原始特征域对应的embedding，在该维度的两两元素和。代码参考：

```
# ---------- FM 二阶特征 ---------------
# 每个域对应的embeeding，乘上自己的特征取值后，开始按元素22交互。
# 每个域的22向量内积<vi,vj>： [a,b,c] * [d,e,f],   [a,b,c] * [h,i,j]
# 原来每个域的embedding是(B，n，K)

# sum_of_square  （1，k）
self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  #  所有特征对应的n个embedding相加：e_sum               
self.summed_features_emb_square = tf.square(self.summed_features_emb)  # e_sum 每个元素平方：相当于[(a+d+h）^2, (b+e)^2,...]

# square_of_sum  （1，k）
self.squared_features_emb = tf.square(self.embeddings)        # 每个特征embedding本身的平方。还是n，K
self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # 平方后的n个embedding相加[a^2+d^2+h^2, b^2+e^2] 

# 相减得到一个K维向量：每个位置对应所有域的向量22内积后，在该位置处对应内积的和。如果直接sum,就是标准的原论文公式：每个域对应向量22内积的和
# 现在不直接相加，而是看做更细粒度的内容：每一维可看做特征域embedding v的该隐藏维度，和其他所有特征域的该维度充分交互后的结果。（bi-interation）
# 如果不同域的embedding，每个维度对应的含义相同。就可以看作是特征在该隐维度的交互。直接sum了就是每个特征域对应向量22内积的和，是标准FM了
# [ad+dh+ah, be+bi+ei,... ]
self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
```



不过在DeepFM里，网上的大多数实现，是用映射得到的所有域embedding拼接后得到的新向量，元素22相乘来得到FM部分的二阶score(Bi-interaction)。样本每个域同样首先映射为对应的embedding。之后的二阶交叉分，是每个域对应的embedding的所有位置两两交互的加和。如果样本的原始特征域是[x1,x2,x3], 对应的embedding是[$\vec{e1}$,$\vec{e2}$,$\vec{e3}$]。那么拼接后的向量是$e$=[$\vec{e1}$,$\vec{e2}$,$\vec{e3}$]=[ v1 ，v2， v3| v4， v5， v6 | v7, v8, v9]，那么最终的FM二阶分数是$e$的每个元素22相乘后的加和：
$$
y_{FM-order2}= \sum_{i}^{9}\sum_{j!=i}^{9}v_iv_j
$$
用原公式实现的FM二阶score，只有不同域embeeding对应元素间的交互($<\vec{e_i},\vec{e_j}> =v_1*v_4 + v2*v5 +v3*v6$)。现在相当于计算了不同域embeeding所有元素的所有交互，既包含不同位置的交互($v_1*v_5,v1*v_6$)，也包含该域对应embedding元素内部的交互：($v_1*v_2,v1*v_3$)。

向量$e=[ v_1 ，v_2， v_3,...]$中的元素22相乘求和，可以写作square_of_sum - sum_of_square：
$$
y_{FM-order2}=  \sum_{i}^{9}\sum_{j!=i}^{9}v_iv_j=\frac{1}{2}((\sum_{i}^{l}v_i)^2 - \sum_{i}^{l}v_i^2)\\
             = v_1v_2 + v_1v_3 + v_2v_3 = \frac{1}{2}((v_1+v_2+v_3)^2 -(v_1^2+v_2^2+v_3^2))
$$

