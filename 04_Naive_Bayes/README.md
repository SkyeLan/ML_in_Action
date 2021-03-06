## 朴素贝叶斯

贝叶斯分类是一类分类算法的总称，这类算法均已贝叶斯定理为基础，故统称为贝叶斯分类。
公式如下：
$$
P(B|A)=\frac{P(A|B)P(B)}{P(A)}
$$
该公式最大的优点就是可以忽略AB的联合概率直接求其条件概率分布。
因为整个形式化过程只做最原始、最简单的建设：各特征属性是**条件独立**的：
$$
P(x|y_i)P(y_i) = P(a_1|y_i)P(a_2|y_i).......P(a_n|y_i)P(y_i)
$$
所以称为“朴素”。

**优点**：在数据较少的情况下仍然有效，可以处理多类别问题

**缺点**：对于输入数据的准备方式较为敏感

适用数据类型：标称型数据

