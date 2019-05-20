Marina Moore
Alex Huang

# Problem Statement
Our goal was to predict whether a paper will get citations. To determine this, we vectorize the text of the abstract using Google's trained word2vec model and train a network with the vectorized paper as input.

Mathematically: Each journal article X[j] is a document of words represented by an n<sub>words,j</sub>x n<sub>encoding</sub> where n<sub>words</sub> is the number of words in the journal j, and n<sub>encoding</sub> is the feature dimension in the gensim KeyedVectors encoding. n<sub>words,j</sub> is different for every journal j, so to address this problem, we will pad the abstracts so that the number of words is the same. To do this, let n<sub>maxwords</sub> = max<sub>j</sub> n<sub>words,j</sub>. Then, let \delta n<sub>words,j</sub>=n<sub>maxwords</sub>-n<sub>words,j</sub>. Let u[j] = mean(X[j]) with the shape (1,n<sub>encoding</sub>) be the mean word, on the average of every word vector in document j. Then, we will pad X[j] with a vector of u<sub>x</sub>[j] of length \delta n<sub>words,j</sub>. This will mean that every X[j] will have a shape of n<sub>maxwords</sub>xn<sub>encoding</sub>.

# Data
We got data from the Scopus database using the PyScopus library. We removed any data that did not have the abstract of the paper. For each abstract, we removed stop words and any words not in the corpus for word2vec then vectorized the data using word2vec. After word-vectorizing the abstracts, we found the average vector for each paper and padded the data with this value to get arrays of equal length, as described above. This is followed by a scaling the data down to unit variance and zero mean, and then applying principal component analysis with a proportion of variance of 0.95.

# Approach
We tried two approaches to predicting the number of citations, logistic regression and a support vector classifier with a linear kernel. 

# Other methods tried
We tried using the full text of the paper, but not enough data was available.
Additionally, we tried predicting the number of citations, but a majority of papers we found had 0 citations, so there was not enough data for the model. Instead, we simplified the problem to a binary classification (zero citations or nonzero). 

# Results
