from cv2 import kmeans, split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
ocean_data=pd.read_csv("N:\Machine learning\Algorithms\data-final.csv",delimiter='\t')

# print(ocean_data.shape)
data=ocean_data.copy()

data.drop(data.columns[50:107],axis=1,inplace=True)
data.drop(data.columns[51:],axis=1,inplace=True)
data.drop('country',axis=1,inplace=True)

                      #------------data scaling-------------


data.fillna(value=data.mean(),inplace=True)
scaler=MinMaxScaler()
data=scaler.fit_transform(data)
data=pd.DataFrame(data)
# print(data.isnull().sum())
# print(data.head(20))
print("building started ------------>")
model=KMeans(n_clusters=5)
data_fit=model.fit(data)
pd.options.display.max_columns=10

predictions=data_fit.labels_

data['Clusters']=predictions
# print(data.head)

from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(data)
pca_2d = pca.transform(data)
df_pca = pd.DataFrame(data=pca_2d, columns=['PCA1', 'PCA2'])
df_pca['Clusters'] = predictions

plt.figure(figsize=(10,10))
sns.scatterplot(data=df_pca,x='PCA1', y='PCA2',hue='Clusters',palette='Set2',alpha=0.9)
print("model successful ")
plt.show()