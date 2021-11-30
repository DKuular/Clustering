#!/usr/bin/env python
# coding: utf-8

# ТЕСТОВОЕ ЗАДАНИЕ  
# Digital Line

# Дана таблица клиентов с различной информацией относительно предпочтений покупок, частоты и сумм (For_clustering.xlsx). Необходимо провести сегментацию клиентов для выделения различных групп.

# In[1]:


#загрузим необходимые пакеты
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import IPython
import sklearn
import seaborn as sns 
from IPython.display import display
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score


# In[2]:


#загрузка пакетов
data=pd.read_excel('/Users/deniskuular/Desktop/Tests/Digital Line/for_clustering.xlsx', engine="openpyxl")
df=data.iloc[:,:43]
y=data.iloc[:,44]
df.info()


# Подготовка данных к анализу

# In[3]:


#Подготовка данных к анализу
df=df.drop(['Age_group'], axis=1)

df['Gender']=df['Gender'].replace (['F', 'M'], [0, 1])

df.Gender.fillna(df.Gender.mean(), inplace=True)
df.Age.fillna(df.Age.mean(), inplace=True)


# Построение модели KMeans (n_clusters=3) с предварительной обработков данных методом StandartScaler. Получение оценки adjusted_rand_score.

# In[4]:


features=StandardScaler().fit_transform(df)

kmeans = KMeans(init="k-means++",n_clusters=3)
kmeans.fit(features)

label_pred_kmeans=pd.Series(kmeans.labels_).replace([0,1,2], [1,2,3])
label_true=y

ari_kmeans=adjusted_rand_score(label_true, label_pred_kmeans)
print('Adjusted_rand_score: ', ari_kmeans) 


# Уменьшение количества признаков методом PCA.

# In[5]:


ari=[]

features=StandardScaler().fit_transform(df)

for i in range(2, 42):
    
    features_pca=PCA(n_components=i).fit_transform(features)
    
    kmeans = KMeans(init="k-means++",n_clusters=3)
    kmeans.fit(features_pca)

    label_pred_kmeans=pd.Series(kmeans.labels_).replace([0,1,2], [1,2,3])
    label_true=y

    ari_kmeans=adjusted_rand_score(label_true, label_pred_kmeans)
    ari.append(ari_kmeans)
print(ari)


# In[6]:


plt.style.use("fivethirtyeight")
plt.plot(range(2, 42), ari)
plt.xticks(range(2, 42))
plt.xlabel("PCA n_components")
plt.ylabel("Adjusted_rand_score")
plt.show()


# Судя по графику наилучшее количество компонентов n_components=2

# In[7]:


features=StandardScaler().fit_transform(df)
features_pca=PCA(n_components=2).fit_transform(features)
kmeans = KMeans(init="k-means++",n_clusters=3)
kmeans.fit(features_pca)

label_pred_kmeans=pd.Series(kmeans.labels_).replace([0,1,2], [1,2,3])
label_true=y

ari_kmeans=adjusted_rand_score(label_true, label_pred_kmeans)
print('Adjusted_rand_score: ', ari_kmeans)


# Найдем оптимальное количество кластеров, используя сумму квадратов расстояния до центра кластера (SSE)

# In[9]:


sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    sse.append(kmeans.inertia_)


# In[10]:


plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()


# In[11]:


#Определим точку изгибу в кривой методом KneeLocator
kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kl.elbow #4


# После того как мы нашли оптимальное количество кластеров проведем новую процедуру кластеризации (n_cluster=4). Полученные результаты визуализируем.

# In[12]:


features=StandardScaler().fit_transform(df)
kmeans_ = KMeans(init="k-means++",n_clusters=4).fit(features)
y_pred_full=kmeans.fit_predict(features)


# In[15]:


plt.scatter(features_pca[y_pred_full == 0, 0], features_pca[y_pred_full == 0,1], s=100, c='blue', label = 'C1')
plt.scatter(features_pca[y_pred_full  == 1, 0], features_pca[y_pred_full  == 1,1], s=100, c='red', label = 'C2')
plt.scatter(features_pca[y_pred_full== 2, 0], features_pca[y_pred_full == 2,1], s=100, c='green', label = 'C3')
plt.scatter(features_pca[y_pred_full == 3, 0], features_pca[y_pred_full== 3,1], s=100, c='yellow', label = 'C4')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=300, c='black', label = 'Centroid')
plt.title('Clusters of Clients')
plt.xlabel('Features')
plt.ylabel('Score')
plt.legend()
plt.show()


# In[16]:


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='black', label = 'Centroid')
plt.title('Clusters centers')
plt.xlim (-5, 6,1)
plt.ylim (-3, 4,1)
plt.show()


# На графиках центры кластеров определены некорректно. Это связано с наличием избыточной информации (данные непрезентативны). Для решения этой задачи воспользуемся методом PCA. Укажем значение агрумента n_components=0.95, которое возвращает минимальное количество признаков с сохранением указанной дисперсии. 

# In[17]:


#KMeans(n_cluster=4) $ PCA(n_components=2) 
features=StandardScaler().fit_transform(df)
features_pca=PCA(n_components=0.95).fit_transform(features) 
kmeans = KMeans(init="k-means++",n_clusters=4)
y_pred=kmeans.fit_predict(features_pca)
print("Количество признаков исходного массива: ", features.shape)
print("Количество признаков скорректированного массива: ", features_pca.shape)


# In[18]:


plt.scatter(features_pca[y_pred == 0, 0], features_pca[y_pred == 0,1], s=100, c='blue', label = 'C1')
plt.scatter(features_pca[y_pred == 1, 0], features_pca[y_pred  == 1,1], s=100, c='red', label = 'C2')
plt.scatter(features_pca[y_pred == 2, 0], features_pca[y_pred == 2,1], s=100, c='green', label = 'C3')
plt.scatter(features_pca[y_pred == 3, 0], features_pca[y_pred == 3,1], s=100, c='yellow', label = 'C4')
#plt.scatter(features_pca[y_pred == 4, 0], features_pca[y_pred == 4,1], s=100, c='purple', label = 'C5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=300, c='black', label = 'Centroid')
plt.title('Clusters of Clients')
plt.xlabel('Features_pca')
plt.ylabel('Score')
plt.legend()
plt.show()


# In[19]:


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='black', label = 'Centroid')
plt.title('Clusters centers')
plt.xlim (-5, 6,1)
plt.ylim (-3, 4,1)
plt.show()


# In[20]:


data['Cluster_3']=label_pred_kmeans
data['Cluster_4']=pd.Series(kmeans.labels_).replace([0,1,2,3], [1,2,3,4])
data


# In[ ]:




