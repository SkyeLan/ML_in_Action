## 逻辑回归

主要思想：根据现有数据对分类边界线建立回归公式

优点：计算代价不高，易于理解和实现

缺点：容易欠拟合，分类精度可能不高

适用数据类型：数值型和标称型数据



Regression问题的常规步骤为：  
1. 寻找h函数（即hypothesis）；
2. 构造J函数（损失函数）；
3. 想办法使得J函数最小并求得回归参数（ θ ）



Logistic函数（Sigmoid函数）： 

$$
g(z)=\frac{1}{1+e^{-z}}
$$

对于线性边界形式：
$$
\theta_0+\theta_1x_1+…+\theta_nx_n=\sum_{i=1}^{n}\theta_ix_i=\theta^Tx
$$
**构造预测函数**为：
$$
h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}
$$
函数 $h_\theta(x)$ 的值有特殊意义，它表示取1的概率，因此对于x分类边界为类别1和类别0的概率为：
$$
\begin{align}
P(y=1|x;\theta)&=h_\theta(x)\\
P(y=0|x;\theta)&=1-h_\theta(x)
\end{align}
$$

**cost函数和J函数**如下，基于最大似然估计推得：
$$
Cost(h_\theta(x),y)=\begin{cases}-log(h_\theta(x))&if\; y=1  \\-log(1-h_\theta(x))&if\; y=0\end{cases}
$$

$$
\begin{align}
J(\theta)=&\frac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(x^{(i)}),y^{(i)})\\
=&-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)})))
\end{align}
$$

推导过程如下：  
将概率公式合起来写：
$$
P(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$
取似然函数为：
$$
\begin{align}
L(\theta)=&\prod_{i=1}^{m}P(y^{(i)}|x^{(i)};\theta)\\
=&\prod_{i=1}^{m}(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{(1-y^{(i)})}
\end{align}
$$

对数取似然函数为：
$$
\begin{align}
l(\theta)=&logL(\theta)\\
=&\sum_{i=1}^{m}(y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)})))
\end{align}
$$

最大似然估计就是求使 $l(\theta)$ 取最大值时的 θ ，其实这里可以使用梯度上升法求解，求得的 θ 就是要求的最佳参数。但是，在Andrew Ng的课程中将 $J(\theta)$ 取为下式，即：
$$
J(\theta)=-\frac{1}{m}l(\theta)
$$
因为乘了一个负的系数 $-\frac{1}{m}$ ，所以取 $J(\theta)$ 最小值时的θ为要求的最佳参数。  



**梯度下降法求的最小值**

开始之前插一句，sigmoid函数有个很重要的特性：
$$
g(z)'=g(z)(1-g(z))
$$


θ 更新过程：
$$
\theta_j:=\theta_j-\alpha\frac{\delta}{\delta_{\theta_j}}J(\theta)
$$
其中：
$$
\begin{align}
\frac{\delta}{\delta_{\theta_j}}J(\theta)
=&-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{h_\theta(x^{(i)})}\frac{\delta}{\delta_{\theta_j}}h_\theta(x^{(i)})-(1-y^{(i)})\frac{1}{1-h_\theta(x^{(i)})}\frac{\delta}{\delta_{\theta_j}}h_\theta(x^{(i)}))\\
=&-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{g(\theta^Tx^{(i)})}-(1-y^{(i)})\frac{1}{1-g(\theta^Tx^{(i)})})\frac{\delta}{\delta_{\theta_j}}g(\theta^Tx^{(i)})\\
=&-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}\frac{1}{g(\theta^Tx^{(i)})}-(1-y^{(i)})\frac{1}{1-g(\theta^Tx^{(i)})})g(\theta^Tx^{(i)})(1-g(\theta^Tx^{(i)}))\frac{\delta}{\delta_{\theta_j}}\theta^Tx^{(i)}\\
=&-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}(1-g(\theta^Tx^{(i)}))-(1-y^{(i)})g(\theta^Tx^{(i)}))x_j^{(i)}\\
=&-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-g(\theta^Tx^{(i)}))x_j^{(i)}\\
=&\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
\end{align}
$$
所以 θ 更新过程可以写为：
$$
\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}
$$

### 向量化

在使用向量化之后公式 $J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}(y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)})))$ 可以写为：
$$
J(\theta)=-\frac{1}{m}(y^Tlog(h)+(1-y^T)log(1-h))
$$
公式 $\frac{\delta}{\delta_{\theta_j}}J(\theta)=\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$ 可以写为：
$$
\frac{\delta}{\delta_\theta}J(\theta)=\frac{1}{m}X^T(h-y)
$$
其中 $h=g(\theta^Tx)$ ，由于 $(h-y)$ 维度是 $m*1$ ， $X$ 也是 $m*1$ ，所以需对 $X$ 转置

所以 θ 的更新过程可写为：
$$
\theta_j:=\theta_j-\alpha\frac{1}{m}X^T(h-y)
$$
