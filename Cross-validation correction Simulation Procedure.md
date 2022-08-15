# Cross-validation correction Simulation Procedure

## 1. Phenotype Simulation

1. randomly select 20 Genes from chromosome 1 from 1000 Genomes Project. 

2. For total dataset, select 5 SNPs as strong effects, simulate $\beta_i$ from $N(0, 200)$ and select 1995 SNPs as weak effects, simulate $\beta_i$ from $N(0, 3)$. So there are totally $p = 5 +1995 = 2,000$ SNPs. 

3. And there are $n$ individuals. So with the bias term $X \in \R^{n \times 6}$ is the fixed genotype .  $W \in \R^{n \times 2000}$ is the genotype which contribute to the random effects.

4. Normalize each SNPs 

5. Get the Phenotype as
   $$
   Y_i = \sum_{j = 1}^{p} \beta_j X_{i,j} +\epsilon_i \ \text{where } i = 1, 2,3,\dots,n
   $$
   where  $\epsilon_i \sim ^{i.i.d} N(0, \sigma_e^2)$,  $\sigma_e^2 = \frac{1 - h^2}{h^2} \sigma_g^2$.

   And $\sigma_g^2 = Var(\sum_{j = 6}^{p} \beta_j X_{i,j})$  = 2609.2

   

## 2. Model Test

### 1. Weighted LSE (weighted linear square estimate)

Let the model be
$$
Y_i = \sum_{j = 1}^{p} \beta_j X_{i,j} + u_i +\epsilon_i \ \text{where } i = 1, 2,3,\dots,n
$$
where $(u_1, u_2, \dots, u_n)^\top \sim MVN(0, \sigma_g^2 K ) $ and $\epsilon_i \sim ^{i.i.d} N(0, \sigma_e^2)$, and  $K = 1/p WW^\top$, $\sigma_e^2 = \frac{1 - h^2}{h^2} \sigma_g^2$ . 



> ? how to estimate $\sigma_e \text{ and } \sigma_g$?  Maybe using REML? But it requires the indication of the clusters.

We can write the model as 
$$
Y_i = \sum_{j = 1}^{p} \beta_j X_{i,j} + \epsilon_i^* \ \text{where } i = 1, 2,3,\dots,n \text{ and } \boldsymbol{\epsilon}^{*} = \boldsymbol u + \boldsymbol \epsilon \sim N_{n}(\mathbf{0}, V)
$$
where $V = Var(Y|X\beta) = \sigma^2_g K + \sigma^2_e I$

let $\sigma_e^2 /\sigma_g^2 = \delta$ so $\sigma_e^2 = \delta \sigma_g^2$

So we need to estimate two of  $\sigma_g, \sigma_e \text{or } h$

And Using FAST-LMM, we get  $\hat \sigma_g =1398.75, \hat \sigma_e = 2780.74, \hat h = 0.32$



##### 1. Just using Weighted LSE

so we can get estimate for $\tilde{\boldsymbol{\beta}}:=\left(X^{t} V^{-1} X\right)^{-1} X^{t} V^{-1} \boldsymbol{Y}$

> $(X^{t} V^{-1} X)$ not invertible **That is a problem**  sudo inverse



##### 2. Using Linear Mixed Model 

BLUP for $\tilde u = E(u|Y) = \sigma^2_g K V^{-1}(Y - X\tilde \beta)$

 And $\hat Y = X \tilde \beta + \tilde u$

Now we can define $H$ by

$\hat Y =HY = X\beta + u$

$H = X\left(X^{t} V^{-1} X\right)^{-1} X^{t} V^{-1} + \sigma^2_g K V^{-1}(I - X\left(X^{t} V^{-1} X\right)^{-1} X^{t} V^{-1})$

So we can get the form of 
$$
\begin{aligned}
\hat Y = & HY = X \tilde\beta + u = X\tilde\beta + \sigma^2_g K V^{-1}(Y - X\tilde \beta)\\
 =& \left[ X W+ \sigma^2_g K V^{-1}(\mathbf{I} - X W) \right] Y
\end{aligned}
$$
So the cross-validation form is
$$
\begin{aligned}
\hat Y_{k} = & H_{k, cv}Y_{-k} = \left[ X_k W_{-k}+ \sigma^2_g K_k V^{-1}_{-k,-k}(\mathbf{I}_{-k} - X_{-k} W_{-k}) \right] Y_{-k}
\end{aligned}
$$





##### 3. LASSO  RIDGE   ENET

[3]
$$
Q(\boldsymbol{\beta})=\frac{1}{n}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})+\sum_{j=1}^{p} P_{\lambda}\left(\left|\beta_{j}\right|\right)
$$
Lasso : $\quad P_{\lambda}=\lambda|\beta|$,

Ridge：$\quad P_{\lambda}=\lambda\left(\beta^{2}\right)$,

ENET : $\quad P_{\lambda}=\lambda\left(\alpha|\beta|+(1-\alpha) \beta^{2}\right)$,

> But Still need to working on how to express as $\hat Y = \hat H Y$

BLUP for $\tilde u = E(u|Y) = \sigma^2_g K V^{-1}(Y - X\tilde \beta)$

 And $\hat Y = X \tilde \beta + \tilde u$

Now we can define $H$ by

$\hat Y =HY = X\beta + u$

$H = X\left(X^{t} V^{-1} X\right)^{-1} X^{t} V^{-1} + \sigma^2_g K V^{-1}(\mathbf I - X\left(X^{t} V^{-1} X\right)^{-1} X^{t} V^{-1})$



#### 1.) Ridge 


$$
Q(\boldsymbol{\beta})=\frac{1}{n}(Y-\mathbf{X} \boldsymbol{\beta})^{\prime}(Y-\mathbf{X} \boldsymbol{\beta})+ \lambda\norm{\beta}_2
$$

$$
\hat \beta = \left(X^\top X +n \lambda \mathbf{I} \right)^{-1} X^{\top} Y\\
$$

So the 
$$
\hat Y = H Y = X \left(X^\top X + n\lambda \mathbf{I} \right)^{-1} X^{\top} Y
$$

> **What is the model assumption for $Cov(Y, Y)$**

#### 2.) Ridge + LMM

Let the model be
$$
Y= X\beta + \mathbf u + \mathbf{\epsilon}
$$
where $\mathbf{u} \sim MVN(0, \sigma_g^2 K ) $ and $\mathbf{\epsilon} \sim ^{i.i.d} N(0, \sigma_e^2 \mathbf{I})$, and  $K = 1/p X X^\top$, $\sigma_e^2 = \frac{1 - h^2}{h^2} \sigma_g^2$ . 

Using Weighted LSE
$$
Y = X\beta + \epsilon^*
$$
where $\epsilon^* = \mathbf{u} + \mathbf{\epsilon} \sim MVN(0, \mathbf V)$ and $\mathbf V = \epsilon_g^2 \mathbf K + \epsilon_e^ 2 \mathbf{I}$ 

Since $\mathbf V$ is symmetric, we can write $\mathbf V = R R^\top$ and the model can be written as $R^{-1}Y = R^{-1}X \beta + R^{-1}\epsilon^* \sim MVN(0, R^{-1 }R R^\top R^{-1^\top} = \mathbf{I})$.

Now if we use Ridge Regression to solve it we get the loss function
$$
Q(\boldsymbol{\beta})=\frac{1}{n}(R^{-1}Y-R^{-1} \mathbf{X} \boldsymbol{\beta})^{\top}(R^{-1}Y-R^{-1}\mathbf{X} \boldsymbol{\beta})+ \lambda\norm{\beta}_2
$$
By minimizing the loss function, we can get 
$$
\begin{aligned}
\hat \beta_{\text{Ridge+LMM}} & = \left( X^\top R^{-1^\top}R^{-1}X + \lambda \mathbf{I}\right)^{-1} X^\top R^{-1^\top}R^{-1}Y\\
& = \left(X^{t} V^{-1} X +n \lambda \mathbf{I}\right )^{-1} X^{t} V^{-1} {Y}
\end{aligned}
$$
BLUP for $\tilde u = E(u|Y) = \sigma^2_g K V^{-1}(Y - X\tilde \beta)$

 And $\hat Y = X \tilde \beta + \tilde u$

Now we can define $H$ by

$\hat Y =HY = X\beta + u$

$H = X\left(X^{t} V^{-1} X  +n \lambda \mathbf{I} \right )^{-1} X^{t} V^{-1} + \sigma^2_g K V^{-1}(I - X\left(X^{t} V^{-1} X  +n \lambda \mathbf{I} \right)^{-1} X^{t} V^{-1})$



------

### 3. BSLMM

[2]
$$
\begin{gathered}
\mathbf{y}=1_{n} \mu+\mathbf{X} \tilde{\beta}+\mathbf{u}+\boldsymbol{\varepsilon} \\
\mathbf{u} \sim \mathbf{M V N}_{n}\left(0, \sigma_{b}^{2} \tau^{-1} \mathbf{K}\right), \\
\boldsymbol{\varepsilon} \sim \operatorname{MVN}_{n}\left(0, \tau^{-1} \mathbf{I}_{n}\right) \\
\tilde{\beta}_{i} \sim \pi \mathbf{N}\left(0, \sigma_{a}^{2} \tau^{-1}\right)+(1-\pi) \delta_{0},
\end{gathered}
$$
(hyper-)parameters, $\mu, \tau, \pi, \sigma_{a}$, and $\sigma_{b}$.

- $\mu$ and $\tau^{-1}$ control the phenotype mean and residual variance.
- $\pi$ controls the proportion of $\tilde{\boldsymbol{\beta}}$ values in (6) that are non-zero.
- $\sigma_{a}$ controls the expected magnitude of the (non-zero) $\tilde{\boldsymbol{\beta}}$.
- $\sigma_{b}$ controls the expected magnitude of the random effects $\mathbf{u}$.

> I am Still studying this methods. But it is fit by using MCMC, which is not capable for doing $\hat Y = H_{cv} Y $ 



## Generalization Error

$$
\begin{align}
\widehat{C V}_{c}= &\frac{1}{n}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)^{t}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)+\frac{2}{n}\left[\operatorname{tr}\left(H_{c v} \operatorname{Cov}(\boldsymbol{y}, \boldsymbol{y})\right)-n \boldsymbol{h}_{t e} \operatorname{Cov}\left(\boldsymbol{y}_{t r}, y_{t e}\right)\right]\\
=& \frac{1}{n}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)^{t}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)+\frac{2}{n} \operatorname{tr}\left[H_{c v} \operatorname{Cov}(\boldsymbol{y}, \boldsymbol{y})\right]  \text{,   when }\operatorname{Cov}\left(\boldsymbol{y}_{t r}, y_{t e}\right)=0 
\end{align}
$$

Since 

 $K = 1/p W W^\top$

In our model $Cov(y,y) = Cov(X \beta + u+\epsilon, X \beta + u+\epsilon) = Cov(u, u)+ Var(\epsilon) = \sigma_g^2 /p W W^{\top} + \sigma_e^2 \mathbf{I}$. 

And the $Cov(y_\text{tr}, y_\text{te}) = Cov(u_\text{tr}, u_\text{te}) = \sigma_g^2 /p W_\text{tr} W_\text{te}^{\top}$

Otherwise
$$
\begin{align}
\widehat{C V}_{c}= &\frac{1}{n}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)^{t}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)+\frac{2}{n}\left[\operatorname{tr}\left(H_{c v} \operatorname{Cov}(\boldsymbol{y}, \boldsymbol{y})\right)-n \boldsymbol{h}_{t e} \operatorname{Cov}\left(\boldsymbol{y}_{t r}, y_{t e}\right)\right]\\

& =
\frac{1}{n}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)^{t}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)+\frac{2}{n}\left[\operatorname{tr}\left(H_{c v} (\sigma_g^2 /p W W^{\top} + \sigma_e^2 \mathbf{I})\right)-n\left(h_{te} \sigma_g^2 /p W_\text{tr} W_\text{te}^{\top}\right)\right] \\

&=\frac{1}{n}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)^{t}\left(\boldsymbol{y}-H_{c v} \boldsymbol{y}\right)+\frac{2}{n}\left[\operatorname{tr}\left(H_{c v} \sigma_g^2 /p W W^{\top}\right)-n\left(h_{te}\ \sigma_g^2 /p W_\text{tr} W_\text{te}^{\top}\right)\right] \\

\end{align}
$$




## References



[1] FaST linear mixed models for genome-wide association studies

[2] Article Source: [**Polygenic Modeling with Bayesian Sparse Linear Mixed Models**](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1003264)
Zhou X, Carbonetto P, Stephens M (2013) Polygenic Modeling with Bayesian Sparse Linear Mixed Models. PLOS Genetics 9(2): e1003264. https://doi.org/10.1371/journal.pgen.1003264

[3] Zeng, P., Zhou, X. & Huang, S. Prediction of gene expression with cis-SNPs using mixed models and regularization methods. *BMC Genomics* **18,** 368 (2017). https://doi.org/10.1186/s12864-017-3759-6





1. Restricted LMM  $ X^{\top}X$ may be singular and determinant is 0.
2.  Singular Matrix - using pseudo inverse
3. Ridge Regression, CVc has flaw




$$
\lfloor \frac{n+2p - f}{S}\rfloor  + 1
$$
