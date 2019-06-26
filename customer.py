import pandas as pd 
import numpy as np 
import os 
from google.colab import files 
file = files.upload() 
import matplotlib.pyplot as plt  
import seaborn as sns  
import plotly as py 
import plotly.graph_objs as go 
from sklearn.cluster import KMeans 
//Check for missing data
dataset.isnull().sum() 
//
df = pd.read_csv(r'Mall_Customers.csv') 

print(df.head()) 

df.shape 

df.describe() 

df.dtypes 

df.isnull().sum() 

plt.style.use('fivethirtyeight') 

plt.figure(1 , figsize = (15 , 6)) 

n = 0  

for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']: 

    n += 1 

    plt.subplot(1 , 3 , n) 

    plt.subplots_adjust(hspace =0.5 , wspace = 0.5) 

    sns.distplot(df[x] , bins = 20) 

    plt.title('Distplot of {}'.format(x)) 

plt.show() 
//Scaling
dfsp = pd.concat([df.mean().to_frame(), df.std().to_frame()], axis=1).transpose()         

dfsp.index = ['mean', 'std'] 

df_scaled = pd.DataFrame() 

for c in df.columns:     

       if(c=='gender'):  

            df_scaled[c] = df[c]    else: df_scaled[c] = (df[c] - dfsp.loc['mean', c]) / dfsp.loc['std', c] 

df_scaled.head()
//PCA Analysis
from sklearn.decomposition import PCA 

kMeans = KMeans(n_clusters=5, random_state=1) 

numeric_cols = df._get_numeric_data().dropna(axis=1) 

kmeans.fit(numeric_cols) 

Pca = PCA(n_components=2) 

Res = Pca.fit_transform(numeric_cols) 

plt.figure(figsize=(12,8)) 

plt.scatter(Res[:,0], Res[:,1], c=kmeans.labels_, s=50, cmap='viridis') 

//FEATURE SELECTION 

X= dataset.iloc[:, [3,4]].values 

//DETERMINING K VALUE BY ELBOW METHOD 

from sklearn.cluster import KMeans 

wcss=[] 

for i in range(1,11): 

    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0) 

     k means.fit(X) 

    wcss.append(kmeans.inertia_) 
    
//Hierarchical Clustering 

//Agglomerative 

 

from sklearn.cluster import AgglomerativeClustering agglom = AgglomerativeClustering(n_clusters=5, linkage='average').fit(X) 

X['Labels'] = agglom.labels_ 

plt.figure(figsize=(12, 8)) 

sns.scatterplot(X['Income'], X['Score'], hue=X['Labels'],                 palette=sns.color_palette('hls', 5)) 

plt.title('Agglomerative with 5 Clusters') 

plt.show() 

plt.plot(range(1,11), wcss) 

plt.title('The Elbow Method') 

plt.xlabel('no of clusters') 

plt.ylabel('wcss') 

plt.show() 

plt.title('PCA');
