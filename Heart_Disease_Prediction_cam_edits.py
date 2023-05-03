#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
import warnings
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')


# In[6]:


df = pd.read_csv('heart.csv')
print(df.head())


# In[7]:


print(df.info())


# In[8]:


print(df.describe())


# In[29]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(16,16))
g=sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()


# In[10]:


sns.set_style('whitegrid')
sns.countplot(x='target', data=df, palette='RdBu_r')
plt.show()


# In[31]:


dataset = pd.get_dummies(df, columns = ['sex','cp',
                                       'fbs', 'restecg',
                                       'exang', 'slope',
                                       'ca', 'thal'])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
print(dataset.head())


# In[15]:


y = dataset['target']
x = dataset.drop(['target'], axis = 1)

from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score = cross_val_score(knn_classifier, x, y, cv=10)
    knn_scores.append(score.mean())


# In[17]:


plt.plot([k for k in range(1,21)], knn_scores, color = 'red')
for i in range(1, 21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number Of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier Scores For Different K Values')
plt.show()


# In[19]:


knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score = cross_val_score(knn_classifier, x, y, cv=10)
score.mean()


# In[20]:


from sklearn.ensemble import RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators=10)
score = cross_val_score(randomforest_classifier, x, y, cv=10)
score.mean()


# In[ ]:




