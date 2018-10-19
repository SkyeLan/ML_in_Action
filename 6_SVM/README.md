## 支持向量机SVM

定义在特征空间上的**间隔最大**的线性分类器，可以通过**核技巧**，变成非线性分类器。

**优点**：泛化错误率低，计算开销不大，结果易解释

**缺点**：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于二分类问题

**适用数据类型**：数值型和标称型数据



点到超平面的距离：
$$
distance = \frac{|w^Tx + b|}{||w||}
$$

找到具有最小间隔的数据点，再对该间隔最大化：
$$
\mathop{\arg\max}_{(w,b)}\{\min_n \ label·\frac{(w^Tx+b)}{||w||}\}
$$

转化后的优目标函数：
$$
\begin{align}
\max_\alpha[\sum_{i=1}^m\alpha-\frac{1}{2}\sum_{i,j=1}^{m}&label^{(i)}·label^{(j)}·\alpha_i·\alpha_j·<x^{(i)},x^{(j)}>] \\\\
st. \ &\alpha\geq0\\
&\sum_{i-1}^m\alpha_i·label^{(i)}=0
\end{align}
$$

引入松弛变量后约束条件变为：
$$
\begin{align}
st. \ &C\geq\alpha\geq0\\&\sum_{i-1}^m\alpha_i·label^{(i)}=0
\end{align}
$$
