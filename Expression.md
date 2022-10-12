Feature & External Feature
Definiation：定义f为一个（数据集D的）固有特征，
$$当且仅当\forall (x,y) \in X \times Y时,(x,y)\in D \Rightarrow (x,y)包含f$$
类似的，定义External Feature $f_e$为
$$ 当且仅当\forall (x,y) \in X \times Y 时,(x,y)包含f_e \Rightarrow (x,y) \notin D$$

Embedding Feature
考虑一个K-分类问题，对每一个样本$(x,y)$,DI首先生成其到每一个类$t$的最小距离$\delta t$
$$ \min_{\delta t} d(x, x+\delta t),s.t. V(x+\delta t)=t $$
$d()$表示距离的度量，到每一个类的距离$\delta=(\delta_1,...,\delta_k)$就是对该样本的Embedding Feature 