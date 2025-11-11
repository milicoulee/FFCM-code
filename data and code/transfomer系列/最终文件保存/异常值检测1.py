import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

new_working_directory = r"D:\毕业论文\傅立叶卷积_itransformer\傅立叶卷积_itransformer\transfomer系列"
os.chdir(new_working_directory)
# 1. 读取数据
df = pd.read_excel('./数据集/风电2019.xlsx')

# 确保时间列是日期类型
df['date'] = pd.to_datetime(df['date'])

# 按日期排序
df = df.sort_values(by='date')

# 提取用于异常检测的列
data = df[['windspeed_10', 'windspeed_30', 'windspeed_50']]

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 2. 孤立森林检测异常点
iso_forest = IsolationForest(contamination= 0.012, random_state=42)
iso_labels = iso_forest.fit_predict(data_scaled)  # -1 表示异常, 1 表示正常
iso_decision = iso_forest.decision_function(data_scaled)  # 异常分数

# 统计孤立森林检测出的异常点数量
iso_anomalies_count = np.sum(iso_labels == -1)
print(f"孤立森林检测出的异常点数量: {iso_anomalies_count}")

# 3. 密度聚类检测异常点
dbscan = DBSCAN(eps=0.08, min_samples=4)
dbscan_labels = dbscan.fit_predict(data_scaled)  # -1 表示异常

# 统计密度聚类检测出的异常点数量
dbscan_anomalies_count = np.sum(dbscan_labels == -1)
print(f"密度聚类检测出的异常点数量: {dbscan_anomalies_count}")

# 4. 可视化异常点（PCA 降维到 2D）
# PCA 降维
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data_scaled)

# 孤立森林异常点可视化
plt.figure(figsize=(12, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=(iso_labels == -1), cmap='coolwarm', label='Isolation Forest Anomaly', alpha=0.6)
plt.title("Isolation Forest Anomaly Detection (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Anomaly (-1: Yes, 1: No)")
plt.legend()
plt.show()

# 密度聚类异常点可视化
plt.figure(figsize=(12, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=(dbscan_labels == -1), cmap='coolwarm', label='DBSCAN Anomaly', alpha=0.6)
plt.title("DBSCAN Anomaly Detection (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label (-1: Anomaly)")
plt.legend()
plt.show()
#%% 分开画图
# import matplotlib.pyplot as plt

# # 假设 data_2d 是通过 PCA 降维后的数据，iso_labels 和 dbscan_labels 是分别来自孤立森林和DBSCAN的标签

# # 创建一个画布并指定子图数量（1行2列）
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # 绘制孤立森林的异常点（左子图）
# sc1 = ax1.scatter(data_2d[:, 0], data_2d[:, 1], c=(iso_labels == -1), cmap='coolwarm', label='Isolation Forest Anomaly', alpha=0.6)
# ax1.set_title("Isolation Forest Anomaly Detection (PCA Reduced)")
# ax1.set_xlabel("PCA Component 1")
# ax1.set_ylabel("PCA Component 2")
# ax1.legend()

# # 为孤立森林的子图添加颜色条
# fig.colorbar(sc1, ax=ax1, label="Anomaly (-1: Yes, 1: No)")

# # 绘制DBSCAN的异常点（右子图）
# sc2 = ax2.scatter(data_2d[:, 0], data_2d[:, 1], c=(dbscan_labels == -1), cmap='winter', label='DBSCAN Anomaly', alpha=0.6)
# ax2.set_title("DBSCAN Anomaly Detection (PCA Reduced)")
# ax2.set_xlabel("PCA Component 1")
# ax2.set_ylabel("PCA Component 2")
# ax2.legend()

# # 为DBSCAN的子图添加颜色条
# fig.colorbar(sc2, ax=ax2, label="Anomaly (-1: Yes, 1: No)")

# # 显示图形
# plt.tight_layout()
# plt.show()
#%%
# 同时可视化两种方法的异常点
plt.figure(figsize=(12, 6))
plt.scatter(data_2d[:, 0], data_2d[:, 1], c='lightgray', label='Normal', alpha=0.5)
plt.scatter(data_2d[iso_labels == -1, 0], data_2d[iso_labels == -1, 1], c='red', label='Isolation Forest Anomaly')
plt.scatter(data_2d[dbscan_labels == -1, 0], data_2d[dbscan_labels == -1, 1], c='blue', label='DBSCAN Anomaly', marker='x')
plt.title("Anomaly Detection Comparison")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.show()







import pandas as pd
import numpy as np

# 假设 df 是原始数据，iso_labels 和 dbscan_labels 是孤立森林和密度聚类的标签
df['iso_label'] = iso_labels  # 孤立森林结果
df['dbscan_label'] = dbscan_labels  # 密度聚类结果

# 1. 找出异常值的位置：孤立森林或密度聚类标记为异常的数据
iso_anomalies = df['iso_label'] == -1  # 孤立森林标记为异常的数据
dbscan_anomalies = df['dbscan_label'] == -1  # 密度聚类标记为异常的数据

# 2. 创建一个副本：将异常值的位置标记为NaN
data_cleaned = df.copy()

# 将被标记为异常的行设置为 NaN
data_cleaned[iso_anomalies | dbscan_anomalies] = np.nan

# 3. 使用线性插值填充缺失值（异常值）
data_cleaned = data_cleaned.interpolate(method='linear', axis=0, limit_direction='both')

# 4. 输出原始数据和清洗后数据的对比
print("原始数据大小:", df.shape)
print("清洗后数据大小:", data_cleaned.shape)
print("前几行清洗后的数据:")
print(data_cleaned.head())

# 5. 将清洗后的数据保存
data_cleaned.to_csv('./数据集/cleaned_data.csv', index=False)





#%%
# 最近邻计算
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
k = 4  # k 表示第 k 个最近邻
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(data_scaled)
distances, indices = neighbors_fit.kneighbors(data_scaled)

# 排序距离
distances = np.sort(distances[:, -1])  # 取每个点的第 k 个最近邻距离

# 绘制最近邻距离图
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 假设 k 已经定义并且 distances 已经计算好
# k = 5  # 例如，k 是 5 表示第5近邻
# distances 是存储最近邻距离的数组（已经计算好的）

# 设置 eps 值
eps = 0.08

# 绘制最近邻距离图
plt.figure(figsize=(10, 6))
plt.plot(distances, color='blue', label=f'{k}th Nearest Neighbor Distance')
plt.axhline(y=eps, color='red', linestyle='--', label=f'eps = {eps}')  # 绘制 eps 线

# 标记异常点：假设异常点的距离大于 eps
outliers = np.where(distances > eps)[0]
plt.scatter(outliers, distances[outliers], color='red', label='Anomalies', zorder=5)

# 设置图形标题、标签和网格
plt.title('k-Distance Graph (for Determining eps)')
plt.xlabel('Data Points Sorted by Distance')
plt.ylabel(f'{k}th Nearest Neighbor Distance')
plt.legend()
plt.grid()

# 显示图形
plt.show()

# 打印 90th Percentile 的值
print("k-Distance Graph Analysis:")
print(f"90th Percentile of {k}th Nearest Neighbor Distance: {np.percentile(distances, 90):.4f}")


