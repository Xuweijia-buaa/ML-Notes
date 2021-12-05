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
所以第一项是$\vec{x}_{(1n)}*\vec{V}_{(nk)}$矩阵相乘后的所有值相加：
$$
\sum_{k}( < \vec{x},\vec{V[k]}>^2)\\
=sum( \vec{x}_{(1n)}*\vec{V}_{(nk)})
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
=   \frac{1}{2}  (sum( \vec{x}*V) - sum( \vec{x^2}*V^2) )
$$
最终的复杂度是每个维度的向量内积<x,V[k]>的$O(n)$，以及最后的k维加和：$O(kn)$