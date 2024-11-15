#读取手写数字书别.gz文件
import gzip
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
from tqdm import tqdm

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



# 定义不同的 perplexity 值
perplexity_values = [5, 10, 20, 40, 80, 150]

# 存储结果和时间
results = {}
times = {}

all_start_time = time.time()
for i, perplexity, in tqdm(enumerate(perplexity_values), total=len(perplexity_values), desc="t-SNE Progress"):
    start_time = time.time()
    
    # 应用 t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_normalized)
    
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 记录结果和时间
    results[perplexity] = X_tsne
    times[perplexity] = elapsed_time


all_end_time = time.time()
all_elapsed_time = all_end_time - all_start_time
print(f"t-SNE time = {all_elapsed_time:.4f}s")

save_dir = f"./task4/time={all_elapsed_time:.4f}s"
os.makedirs(save_dir, exist_ok=True)

# 创建一个图形窗口
fig, axes = plt.subplots(2, 3, figsize=(15, 10))


# 遍历不同的最近邻数
with open(os.path.join(save_dir, f"t-SNE Calculate time.txt"), 'w') as file:
    for i, perplexity in enumerate(perplexity_values):
        ax = axes[i // 3, i % 3]
        unique_labels = np.unique(labels_sample)
        for label in unique_labels:
            ax.scatter(results[perplexity][labels_sample == label, 0], results[perplexity][labels_sample == label, 1], label=str(label), s=10)
        ax.set_xlabel("t-SNE componet1")
        ax.set_ylabel("t-SNE componet1")
        ax.legend()
        ax.set_title(f"perplexity:{perplexity}")
        
        text = f"perplexity: {perplexity}     计算时间: {times[perplexity]:.4f} 秒\n"
        file.write(text)

fig.suptitle('t-SNE of dataset with different k values', fontsize=16, y=1.03)
plt.savefig(os.path.join(save_dir,  "t-SNE of dataset with different k values.png"))
plt.close()