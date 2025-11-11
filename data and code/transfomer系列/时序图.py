import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('data.csv')
data['date'] = pd.to_datetime(data['date'])  # 确保日期为 datetime 格式

# 添加年份列
data['year'] = data['date'].dt.year

# 筛选需要的年份 (2017, 2018, 2019)
selected_years = [2017, 2018, 2019]
filtered_data = data[data['year'].isin(selected_years)]

# 获取筛选后的年份
unique_years = filtered_data['year'].unique()

# 设置画布大小和子图的行数
num_years = len(unique_years)
fig, axes = plt.subplots(num_years, 1, figsize=(18, 3 * num_years))  # 每行一个子图

# 如果只有一个年份，axes不会是数组，需要特殊处理
if num_years == 1:
    axes = [axes]

# 为每一年绘制子图
for i, year in enumerate(unique_years):
    yearly_data = filtered_data[filtered_data['year'] == year]
    axes[i].plot(yearly_data['date'], yearly_data['Target'], label=f'Target ({year})', color='blue')
    axes[i].set_title(f'Time Series Plot of Target Variable for {year}')
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('Target')
    axes[i].legend()
    axes[i].grid()
    axes[i].set_xlim(yearly_data['date'].min(), yearly_data['date'].max())  # 设置独立的 x 轴范围

# 调整子图布局
plt.tight_layout()
plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取数据
# data = pd.read_csv('data.csv')

# # 删除非数值列（如日期列）
# numeric_data = data.select_dtypes(include=['float64', 'int64'])

# # 绘制箱线图
# plt.figure(figsize=(12, 8))
# numeric_data.boxplot()
# plt.title('Boxplot of All Variables')
# plt.xticks(rotation=45)  # 旋转 x 轴标签以便更好显示
# plt.xlabel('Variables')
# plt.ylabel('Values')
# plt.grid(False)  # 去除网格线（可根据需求调整）
# plt.show()

# import pandas as pd

# # 读取数据
# data = pd.read_csv('data.csv')

# # 删除非数值列（如日期列）
# numeric_data = data.select_dtypes(include=['float64', 'int64'])

# # 描述性统计分析
# stats = numeric_data.describe().T  # 转置方便查看，行是变量，列是统计指标

# # 添加自定义统计指标：中位数
# stats['median'] = numeric_data.median()

# # 重命名行列名便于理解
# stats = stats.rename(columns={
#     'mean': 'Average',
#     'std': 'Standard Deviation',
#     'min': 'Minimum',
#     '25%': '25th Percentile',
#     '50%': '50th Percentile',
#     '75%': '75th Percentile',
#     'max': 'Maximum'
# })

# # 查看结果


