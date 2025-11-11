import pandas as pd
import matplotlib.pyplot as plt

# 假设 df 是你的 DataFrame，包含 'date' 列以及其他特征列
df = pd.read_csv('./data.csv')  # 读取数据
df['date'] = pd.to_datetime(df['date'])  # 确保日期列是日期时间格式

# 定义季节划分函数
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

# 添加季节列
df['season'] = df['date'].dt.month.apply(get_season)

# 提取2017年的数据
df_2017 = df[df['date'].dt.year == 2017]

# 按季节分割数据，并重置索引
seasonal_data_2017 = {
    'Winter': df_2017[df_2017['season'] == 'Winter'].reset_index(drop=True),
    'Spring': df_2017[df_2017['season'] == 'Spring'].reset_index(drop=True),
    'Summer': df_2017[df_2017['season'] == 'Summer'].reset_index(drop=True),
    'Autumn': df_2017[df_2017['season'] == 'Autumn'].reset_index(drop=True)
}

# 绘制每个季节的时序图
plt.figure(figsize=(14, 10))

# 绘制冬季数据
plt.subplot(2, 2, 1)
plt.plot(seasonal_data_2017['Winter'].index, seasonal_data_2017['Winter']['Target'], color='tab:blue', label='Windspeed 100m')
plt.title('2017 Winter: Windspeed 100m')
plt.xlabel('Index')
plt.ylabel('Windspeed (m/s)')
plt.legend()

# 绘制春季数据
plt.subplot(2, 2, 2)
plt.plot(seasonal_data_2017['Spring'].index, seasonal_data_2017['Spring']['Target'], color='tab:green', label='Windspeed 100m')
plt.title('2017 Spring: Windspeed 100m')
plt.xlabel('Index')
plt.ylabel('Windspeed (m/s)')
plt.legend()

# 绘制夏季数据
plt.subplot(2, 2, 3)
plt.plot(seasonal_data_2017['Summer'].index, seasonal_data_2017['Summer']['Target'], color='tab:red', label='Windspeed 100m')
plt.title('2017 Summer: Windspeed 100m')
plt.xlabel('Index')
plt.ylabel('Windspeed (m/s)')
plt.legend()

# 绘制秋季数据
plt.subplot(2, 2, 4)
plt.plot(seasonal_data_2017['Autumn'].index, seasonal_data_2017['Autumn']['Target'], color='tab:orange', label='Windspeed 100m')
plt.title('2017 Autumn: Windspeed 100m')
plt.xlabel('Index')
plt.ylabel('Windspeed (m/s)')
plt.legend()

plt.tight_layout()  # 自动调整子图之间的间距
plt.show()
