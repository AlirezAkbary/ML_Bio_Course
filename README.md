# Machine Learning for Bioinformatics
My Solutions for Machine Learning for Bioinformatics Course (Graduate Course) Assignments and Research Project.


## Project: Drug-Protein Affinity Classification
The project is about the prediction of binding between proteins and drugs. In phase 1, this is done by machine learning algorithms (XGBoost).


In phase 2,  the problem was to improve the limitations of the well-known model DeepDTA. By reading the state-of-the-art paper GraphDTA, which takes advantage of Graph Neural Networks, I modified DeepDTA by implementing LSTM to learn protein sequence (as DeepDTA doesn't take the sequential nature of target amino-acid structures into account) and graph convolutional network to learn drug structure. Also, I applied some interpretability methods to analyze the network learned on data and got valuable insights that the learned model is overly dependent on the drugs without a reasonable focus on the proteins. 


My literature review on the topic consisted:
+ **DeepDTA: Deep Drug-Target Binding Affinity Prediction** [(arxiv)](https://arxiv.org/pdf/1801.10193.pdf)
+ **GraphDTA: prediction of drug–target binding affinity using graph convolutional networks** [(bioarxiv)](https://www.biorxiv.org/content/10.1101/684662v3.full.pdf)
+ **DeepGS: Deep Representation Learning of Graphs and Sequences for Drug-Target Binding Affinity Prediction** [(arxiv)](https://arxiv.org/pdf/2003.13902.pdf)
+ Saliency Maps DNN Interpretation or: **Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps** [(arxiv)](https://arxiv.org/pdf/1312.6034.pdf)
+ Guided Back Propagation DNN Interpretation or:**STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET** [(arxiv)](https://arxiv.org/pdf/1412.6806.pdf)
+ LRP DNN Interpretation or:**On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation** [(arxiv)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
+ Microsoft Research Tutorials on Graph Neural Networks. [(link)](https://youtu.be/zCEYiCxrL_0), [(link)](https://youtu.be/cWIeTMklzNg)
+ PyTorch Geometric Extension Library. [(link)](https://pytorch-geometric.readthedocs.io/en/latest/)

## Homeworks

### HW6
Covered topics:
+ Autoencoders
+ VAE (Theory & Implementation) [(arxiv)](https://arxiv.org/pdf/1312.6114.pdf)
+ GAN [(arxiv)](https://arxiv.org/pdf/1406.2661.pdf), Wasserstein GAN, Mode Collapse and Mini Batch Discrimination [(arxiv)](https://arxiv.org/pdf/1701.00160.pdf), [(link)](https://papers.nips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)
+ RNN, LSTM (Theory & Implementation)


### HW5
Covered topics:
+ Hidden Markov Models
+ Deep Learning Basics (Also a more rigorous view on Batch Normalization by the paper: **How Does Batch Normalization Help Optimization?** [(arxiv)](https://arxiv.org/pdf/1805.11604.pdf) and SGD Optimization in Over-parameterized Network by: **A Convergence Theory for Deep Learning via Over-Parameterization** [(arxiv)](https://arxiv.org/pdf/1811.03962.pdf))
+ Universal Approximation of Neural Networks 
+ MLP Implementation from Scratch
+ Reading and Implementation of ResNet Paper with PyTorch.[(arxiv)](https://arxiv.org/pdf/1512.03385.pdf) Also a more rigorous view on ResNet by the papers: **Visualizing the Loss Landscape of Neural Nets** [(arxiv)](https://arxiv.org/pdf/1712.09913.pdf) and **Deep Residual Networks, Deep Learning Gets Way Deeper** by Kaiming He.[(link)](https://icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf)


### HW4
Covered topics:
+ PCA (Theory & Implementation), ICA
+ K-Means
+ GMM (Theory & Implementation), Expectation Maximization and Variational Lower Bound
+ Reading t-SNE paper [(link)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)


### HW3
Covered topics:
+ Ensemble Learning, Bagging, Boosting such as Random Forest, AdaBoost (Theory & Implementation)
+ Feature Selection (Bayesian Networks, Markov Blanket, and d-separation - LASSO Regularizer)


### HW2
Covered topics:
+ Perceptron (Theory & Implementation)
+ Support Vector Machine (Theory & Implementation)
+ Kernel Methods

### HW1
Covered topics:
+ Basics of Information Theory
+ Decision Tree (Theory & Implementation)
+ KNN (Theory & Implementation)
+ Hypothesis Testing of The Performance of the Models


### HW0
Covered topics:
+ Review of Multivariable calculus
+ Review of Linear Algebra
+ Review of Probability & Statistics

