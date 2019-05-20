Marina Moore
Alex Huang

# Problem Statement
Our goal was to predict whether a paper will get citations. To determine this, we vectorize the text of the abstract using Google's trained word2vec model and train a network with the vectorized paper as input.

Mathematically: Each journal article X[j] is a document of words represented by an n<sub>words,j</sub>x n<sub>encoding</sub> where n<sub>words</sub> is the number of words in the journal j, and n<sub>encoding</sub> is the feature dimension in the gensim KeyedVectors encoding. n<sub>words,j</sub> is different for every journal j, so to address this problem, we will pad the abstracts so that the number of words is the same. To do this, let n<sub>maxwords</sub> = max<sub>j</sub> n<sub>words,j</sub>. Then, let \delta n<sub>words,j</sub>=n<sub>maxwords</sub>-n<sub>words,j</sub>. Let u[j] = mean(X[j]) with the shape (1,n<sub>encoding</sub>) be the mean word, on the average of every word vector in document j. Then, we will pad X[j] with a vector of u<sub>x</sub>[j] of length \delta n<sub>words,j</sub>. This padding makes it so that every word-vectorized abstract will have the same number of words, i.e. every X[j] will have a shape of n<sub>maxwords</sub>xn<sub>encoding</sub>.

# Data
We got data from the Scopus database using the PyScopus library. We removed any data that did not have the abstract of the paper. For each abstract, we removed stop words and any words not in the corpus for word2vec then vectorized the data using word2vec. This word2vec scheme, implemented in the gensim package in Python, is an embedding that has been pre-trained on a Google dataset, with an appropriate distance metric between any pair of two words in the corpus. After word-vectorizing the abstracts, we found the average vector for each paper and padded the data with this value to get arrays of equal length, as described above. This is followed by a scaling the data down to unit variance and zero mean, and then applying principal component analysis with a proportion of variance of 0.95.

At this point, the one main issue that still remains is that the dataset is imbalanced; the y=0 class consists of more than 80% of the data, leading to very few data for the y=1 class. To do this, we tried upsampling techniques, as well as setting the class_weight hyperparameter in the classification algorithms to 'balanced'.

# Approach
We tried two approaches to predicting the number of citations, logistic regression and a support vector classifier with a linear kernel. Should time have allowed, we would have ran a hyperparameter optimization via GridSearchCV, using 5 KFolds and F1-score as the metric (reason described below).

# Other methods tried
We tried using the full text of the paper to predict the actual number of citations using regression, but not only was there insufficient data, but the dataset is extremely imbalanced, as described above.
Therefore, we simplified the problem by just using the abstract, and framing the task as a binary classification (zero citations or nonzero). 

# Results
Since the dataset is imbalanced, we will use the F1-score as the scoring metric instead of accuracy. This is because you can get a very high accuracy rate just by classifying everything as yhat = 0! Our F1-score came out to be very poor: 0.3. There is much further work that can be done on this prediction problem, including collecting more data, cleaning the data better, evaluating how much signal is in the data in the first place (by permuting y-vector and seeing if our classification algorithm performs adequately better compared to random), or just questioning the word embedding scheme we used in the first place (the one described above was very desired because its distance metrics were pre-trained and make sense). Anything in the data processing pipeline, as well as model selection, is fair game for analyzing more deeply.
