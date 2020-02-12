import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  #导入kmeans
from sklearn.utils import shuffle
import numpy as np
from skimage import io
import matplotlib as mpl


import warnings
warnings.filterwarnings('ignore')

original = mpl.image.imread('./images/1.jpg')
width,height,depth = original.shape
temp = original.reshape(width*height,depth)
temp = np.array(temp, dtype=np.float64) / 255

original_sample = shuffle(temp, random_state=0)[:1000] #随机取1000个RGB值作为训练集
def cluster(k):
    estimator = KMeans(n_clusters=k,n_jobs=8,random_state=0)#构造聚类器
    kmeans = estimator.fit(original_sample)#聚类
    return kmeans


def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

kmeans = cluster(32)
labels = kmeans.predict(temp)
#
# print( labels.shape ) # (1555200,)
# print ( labels )# [10 10 10 ... 16  5  5]
kmeans_32 = recreate_image(kmeans.cluster_centers_, labels,width,height)
# print (  kmeans.cluster_centers_.shape )


kmeans = cluster(64)
labels = kmeans.predict(temp)
kmeans_64 = recreate_image(kmeans.cluster_centers_, labels,width,height)

kmeans = cluster(128)
labels = kmeans.predict(temp)
kmeans_128 = recreate_image(kmeans.cluster_centers_, labels,width,height)


plt.figure(figsize = (15,10))
plt.subplot(2,2,1)
plt.axis('off')
plt.title('Original image')
plt.imshow(original )

plt.subplot(2,2,2)
plt.axis('off')
plt.title('Quantized image (128 colors, K-Means)')
io.imsave('kmeans_128.png',kmeans_128)
plt.imshow(kmeans_128)

plt.subplot(2,2,3)
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
io.imsave('kmeans_64.png',kmeans_64)
plt.imshow(kmeans_64)
plt.subplot(2,2,4)
plt.axis('off')
plt.title('Quantized image (32 colors, K-Means)')
plt.imshow(kmeans_32)
plt.show()

