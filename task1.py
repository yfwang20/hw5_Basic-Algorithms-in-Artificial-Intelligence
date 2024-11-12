#读取手写数字书别.gz文件
import gzip
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os

with gzip.open('./MNIST_dataset/train-images-idx3-ubyte.gz', 'rb') as f:
    images = np.frombuffer(f.read(), np.uint8, offset=16)
    images = images.reshape(-1, 784)

with gzip.open('./MNIST_dataset/train-labels-idx1-ubyte.gz', 'rb') as f:
   labels = np.frombuffer(f.read(), np.uint8, offset=8)

#建议设置相同的随机数种子，来让实验结果可以复现
random.seed(20241106)
#完成数据的预处理，每一个数字保留1000个，最后得到10000大小的数据集
labels_sample=[]
images_sample=[]
for i in range(0,10):
    #获得标签为i的索引
    labels_i=[ index for index,label in enumerate(labels) if label==i ]
    # print(f"the amount of number {i} is :",len(labels_i))
    #随机找1000个
    indices_chosen=random.sample(labels_i,1000)
    images_chosen=images[indices_chosen]
    labels_chosen=labels[indices_chosen]
    #找完存起来
    images_sample.append(images_chosen)
    labels_sample.append(labels_chosen)
#揉成一个大的数据集
labels_sample=np.array([item for sublist in labels_sample for item in sublist])
images_sample=np.array([item for sublist in images_sample for item in sublist])
'''
思考：目前这样采数据，所有标签为1的都会在[1000,1999]，标签为3的都会在[3000.3999]，而没有打乱，
这样会不会影响最终的降维结果？
'''
#如果觉得有影响的话，可以打乱顺序：
'''
indices=np.arange(len(labels_sample))
np.random.shuffle(indices)
images_sample=images_sample[indices]
labels_sample=labels_sample[indices]
'''
#标准化，思考：为什么要标准化？
EX=images_sample.mean(axis=0)
sqrt_DX=images_sample.std(axis=0)
X_normalized=(images_sample-EX)/(sqrt_DX+1e-10)

def plot_with_annotation(X , labels , X_name='X axis' , Y_name='Y axis' , graph_name='2d graph', save_dir = './task1'):
    '''
    参数说明：
    X：数据矩阵（降维后的）
    labels：标签矩阵
    X_name:X轴的名字
    Y_name:Y轴的名字
    graph_name:图像的名字
    '''
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(X[labels == label, 0], X[labels == label, 1], label=str(label))
    plt.xlabel(X_name)
    plt.ylabel(Y_name)
    plt.legend()
    plt.title(graph_name)
    plt.savefig(os.path.join(save_dir, graph_name + ".png"))
    plt.close()

def pca(X_normalized, n_components=2):
    """主成分分析（PCA）"""

    # 1. 计算协方差矩阵
    cov_matrix = np.cov(X_normalized, rowvar=False)
    
    # 2. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 3. 选择前 n_components 个特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
    
    # 4. 投影数据
    X_pca = np.dot(X_normalized, top_eigenvectors)
    
    return X_pca, eigenvalues[sorted_indices], top_eigenvectors

start_time = time.time()
X_pca, eigenvalues, top_eigenvectors = pca(X_normalized, n_components=2)
end_time = time.time()
print(f"PCA time = {end_time - start_time:.4f}s")

save_dir = f"./task1/time={end_time - start_time:.4f}s"
os.makedirs(save_dir, exist_ok=True)

plot_with_annotation(X_pca,labels_sample,"PCA componet1","PCA component2",'PCA of dataset', save_dir)

# 绘制前 30 个特征值的曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, 31), eigenvalues[:30], marker='o')
plt.title('Top 30 Eigenvalues')
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.savefig(os.path.join(save_dir, "Top 30 Eigenvalues of pca.png"))
plt.close()