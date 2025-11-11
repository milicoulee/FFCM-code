import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('./数据集/风电2019.xlsx')

# 假设数据集中有一个时间列，可以通过将其转换为 datetime 类型来确保正确处理
# 如果时间列的名字是 "Date"，可以调整列名
df['Date'] = pd.to_datetime(df['date'])

# 设置 Date 列为索引
df.set_index('Date', inplace=True)

# 绘制时间序列图
plt.figure(figsize=(12, 6))
plt.plot(df['Target'], label='Wind Power (Target)', color='blue')
plt.title('Wind Power Time Series')
plt.xlabel('Time')
plt.ylabel('Power (MW)')
plt.legend()
plt.grid(True)
plt.show()
