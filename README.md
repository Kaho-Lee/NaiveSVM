# NaiveSVM
A naive reimplementation of support vector machine and optimization algorithm analysis.

Two SVM are developed from scratch:

1. Soft margin binary classification SVM is developed. Sequential minimal optimization and
 Quadratic Penalty method with using Quasi-Newton methods (BFGS) are devloped to training the model

2. An unsupervised variant of SVM, one-class SVM, is developed for anomaly detection application.
And Quadratic Penalty method with Quasi-Newton methods (BFGS) is devloped for model training.

There are several insightful referrence which help this project a lot

1. https://cling.csd.uwo.ca/cs860/papers/SVM_Explained.pdf (A brief mathematical derivation of SVM)

2. http://cs229.stanford.edu/materials/smo.pdf (A brief introduction of SMO in SVM, pseudocode provided)

3. https://link.springer.com/book/10.1007/978-0-387-40065-5 (Main referrence of optimization algorithms)

Three experiment scripts are included.
1. experiment.m is script to reimplement the experiment in Matlab SVM documentation

2. Ionosphere_BinaryClassification.m solves the binary classification problem for Ionosphere dataset
from UCI machine learning repository https://archive.ics.uci.edu/ml/datasets/Ionosphere

3. Ionosphere_AnomalyDetection.m is to simulate the anomaly detection of the developed model using
Ionosphere dataset.