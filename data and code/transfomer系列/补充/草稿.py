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
lower_bound = Q1 - 1.8 * IQR
upper_bound = Q3 + 1.8 * IQR

# 找出每列的异常值
outliers = (data < lower_bound) | (data > upper_bound)

# 计算每列异常值的数量
outlier_count_per_column = outliers.sum()

# 输出每列异常值的数量
print("每列的异常点数量：")
print(outlier_count_per_column)

# 计算异常值的比例
total_points = 35040
outlier_ratio = outlier_count_per_column / total_points

# 输出每列的异常值比例
print("每列的异常值比例：")
print(outlier_ratio)

import seaborn as sns



# 设置图表的大小
plt.figure(figsize=(12, 8))

# 创建子图，按列分别绘制箱线图
for i, column in enumerate(data.columns, 1):
    plt.subplot(2, 2, i)  # 2行2列的子图
    sns.boxplot(x=data[column], color='skyblue')
    plt.title(f'Boxplot of {column}')
    plt.xlabel(column)

# 调整子图布局，避免重叠
plt.tight_layout()
plt.show()


plt.style.use('seaborn')  # 使用seaborn风格使图形更美观
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建画布
plt.figure(figsize=(12, 6), dpi=150)

# 绘制箱线图
data.boxplot(grid=True,  # 显示网格线
            vert=True,  # 垂直箱线图
            patch_artist=True,  # 填充颜色
            boxprops={'color':'black','facecolor':'#8FAADC'},  # 箱体样式
            flierprops={'marker':'o','markerfacecolor':'red','markersize':3},  # 异常点样式
            medianprops={'linestyle':'-','color':'yellow'})  # 中位线样式

# 添加标题和标签
plt.title('风电数据特征分布与异常值检测', fontsize=14, pad=20)
plt.xlabel('气象特征', fontsize=12)
plt.ylabel('数值范围', fontsize=12)
plt.xticks(rotation=45)  # X轴标签旋转45度

# 保存高清图片（可选）
plt.savefig('boxplot_analysis.png', bbox_inches='tight', dpi=300)

# 显示图形
plt.tight_layout()
plt.show()
    

#%%
# 创建带两个子图的画布（1行2列横向排列）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
plt.subplots_adjust(wspace=0.3)  # 调整子图间距

# ========== 风速子图 ==========
# 提取风速数据
wind_data = data[['windspeed_10', 'windspeed_30', 'windspeed_50']]

# 绘制风速箱线图（左轴）
wind_boxes = ax1.boxplot(
    wind_data.values,
    labels=wind_data.columns,
    patch_artist=True,
    boxprops={'color':'black','facecolor':'#8FAADC'},
    flierprops={'marker':'o','markerfacecolor':'red','markersize':3},
    medianprops={'linestyle':'-','color':'yellow'},
    whis=2  # 设置胡须长度为2倍IQR
)

# 设置风速子图属性
ax1.set_title('风速特征分布', fontsize=12, pad=15)
ax1.set_ylabel('风速 (m/s)', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='x', rotation=0)

# ========== 功率子图 ==========
# 绘制Target箱线图（右轴）
target_box = ax2.boxplot(
    data['Target'],
    labels=['Target'],
    patch_artist=True,
    boxprops={'color':'black','facecolor':'#6DA85F'},  # 使用不同颜色区分
    flierprops={'marker':'o','markerfacecolor':'darkred','markersize':3},
    medianprops={'linestyle':'-','color':'gold'},
    whis=2  # 设置胡须长度为2倍IQR
)

# 设置功率子图属性
ax2.set_title('功率特征分布', fontsize=12, pad=15)
ax2.set_ylabel('功率 (MW)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.tick_params(axis='x', rotation=0)

plt.show()


# ========== 全局设置 ==========


# 保存图片
plt.savefig('dual_axis_boxplot.png', bbox_inches='tight', dpi=300)
plt.show()
