#!/usr/bin/env python
# coding: utf-8

# In[13]:


#iris flower project 
#objective is to find the species of 
#the particular iris flower.


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
iris = pd.read_csv("iris_dataset.csv")


# In[15]:


#what is the shape of dataset?
iris.shape


# In[16]:


#how many columns are there in dataset and what are the name of column ?
iris.columns


# In[20]:


#how many flower of each species are present?
iris['species'].value_counts()
#iris datasets are balanced as number of flowers of each species are same


# In[28]:


iris.plot(kind = 'scatter', x = 'sepal_length', y = 'sepal_width');
plt.grid()
plt.show()


# In[32]:


#here 'sns' corresponds to seaborn
sns.set_style("whitegrid")
sns.FacetGrid(iris, hue="species", height=4).map(plt.scatter,"sepal_length","sepal_width").add_legend();
plt.show()

#notice that the blue points can be separated easily 
#from red and green by drawing a line.
#but red and green datapoints can not be separated easily


# In[34]:


#3-D  plot
#not recommended generally because 
#it requires very much mouse movement
#in case of 3D instead of sepal_length and sepal_width
#we will also consider petal length
#https://pypi.org/project/pyplotter/   visit this page for better understating or just type 'plotly' 3-D plotting


# In[38]:


#pairwise scatter plot: Pair-Plot
#only visible in 2D
plt.close();
sns.set_style("whitegrid");
sns.pairplot(iris, hue="species", height=3);
plt.show()
#ignore diagonal element/graph for a while
#4c2 = 6  so no of plotting will be 6


#  Observations:
# 1.petal_length and petal_width are the most useful      features to classify various flower types.
# 
# 2.While Setosa can be easily identified (linearly separable), Virnica and Versicolor have some overlap(almost linearly separable).

# In[ ]:




