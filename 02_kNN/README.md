kNN原理：对新来的样本计算样本的每个特征与数据集中对应的特征之间的距离，选取前 k 个距离最小的样本，根据他们的标签使用投票法确定新来样本的标签。

优点：精度高、对异常值不敏感、无数据输入假定

缺点：计算复杂度高、空间复杂度高

使用数据范围：数值型与标称型
