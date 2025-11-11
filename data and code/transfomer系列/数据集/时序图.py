import pandas as pd
import matplotlib.pyplot as plt

# 文件路径
files = [
    './Location1_hd.csv',
    './Location2_hd.csv',
    './Location3_hd.csv',
    './Location4_hd.csv'
]

# 加载数据并提取前8000个数据点的 Target 列
data = [pd.read_csv(file)['windspeed_100m'][:10000] for file in files]

# 创建图表
plt.figure(figsize=(14, 10))

# 为每个地点的数据绘制时序图
for i, target_data in enumerate(data):
    plt.subplot(2, 2, i + 1)  # 创建子图
    plt.plot(target_data, label=f'Location {i+1}', linewidth=1)
    plt.title(f'Location {i+1} - Target Time Series')
    plt.xlabel('Index')
    plt.ylabel('Target')
    plt.grid(True)
    plt.legend()

plt.tight_layout()  # 调整布局以避免重叠
plt.show()
