import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

new_working_directory = r"D:\毕业论文\傅立叶卷积_itransformer\傅立叶卷积_itransformer\transfomer系列"
os.chdir(new_working_directory)
# 假设df是你的DataFrame
import pandas as pd
import numpy as np

# 假设 df 是你加载的数据
df = pd.read_excel('./数据集/风电2019.xlsx')

# 确保时间列是日期类型
df['date'] = pd.to_datetime(df['date'])

# 按日期排序
df = df.sort_values(by='date')

# 提取用于异常检测的列
data = df[['windspeed_10', 'windspeed_30', 'windspeed_50', 'Target']]

# 计算 IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# 计算异常值的上下界
lower_bound = Q1 - 2 * IQR
upper_bound = Q3 +2 * IQR

# 找出每列的异常值
outliers = (data < lower_bound) | (data > upper_bound)

# 计算每列异常值的数量
outlier_count_per_column = outliers.sum()

# 输出每列异常值的数量
print("每列的异常点数量：")
print(outlier_count_per_column)

# 可视化异常值检测
plt.figure(figsize=(12, 6))
plt.plot(df['date'], data['windspeed_10'], label='Windspeed 10m', color='tab:red')

# 标记异常值
plt.scatter(df['date'][outliers['windspeed_10']], data['windspeed_10'][outliers['windspeed_10']], color='tab:orange', label='Anomalies', zorder=5)

plt.title('Windspeed 10m with Detected Anomalies')
plt.xlabel('Date')
plt.ylabel('Windspeed (m/s)')
plt.legend()
plt.show()

# 填充异常值：使用前后值的插值
# 这里只处理标记为异常的值，将其填充
data_cleaned = data.copy()
data_cleaned[outliers] = np.nan  # 将异常值设为NaN

# 使用线性插值填充缺失值（异常值）
data_cleaned = data_cleaned.interpolate(method='linear', limit_direction='both')

# 也可以选择使用均值填充：
# data_cleaned = data_cleaned.fillna(data.mean())

# 可视化填充后的结果
plt.figure(figsize=(12, 6))
plt.plot(df['date'], data['windspeed_10'], label='Original windspeed_100m', color='tab:red', alpha=0.5)
plt.plot(df['date'], data_cleaned['windspeed_10'], label='Cleaned windspeed_100m', color='tab:green')
plt.title('Cleaned Windspeed 100m (After Outlier Removal and Filling)')
plt.legend()
plt.show()

# 输出处理后的数据
print(data_cleaned.head())
