import pandas as pd


df = pd.read_csv('./数据集/01.csv')
df.rename(columns={
    'DATATIME': 'date'
}, inplace=True)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)


df['date'] = df['date'].dt.strftime('%Y/%m/%d %H:%M:%S')


df.drop_duplicates(subset = ['date'], keep='first', inplace=True)

df = df.sort_values(by='date', ascending=True)




df = df[~df['date'].str.startswith('2021/10/01')]
#%%
missing_values = df.isna().sum()
print("缺失值统计:")
print(missing_values)
for col in df.columns:
    if col != 'YD15' and df[col].isnull().any():
        df[col].fillna(method='ffill', inplace=True)
missing_values = df.isna().sum()
print("缺失值统计:")
print(missing_values)
# 异常值处理
# 1. 对 YD15 小于 -800 的值进行处理
df.loc[df['YD15'] < -800, 'YD15'] = 0

# 2. 对 WINDSPEED < 1 且 YD15 > 800 的值进行处理
df.loc[(df['WINDSPEED'] < 1) & (df['YD15'] > 800), 'YD15'] = 0
#%%
df.rename(columns={
    'WINDSPEED': '风速',
    'WINDDIRECTION': '风向',
    'TEMPERATURE': '温度',
    'HUMIDITY': '湿度',
    'PRESSURE': '气压',
    'PREPOWER': '预测功率',
    'ROUND(A.WS,1)': '实际风速',
    'ROUND(A.POWER,0)': '实际功率（计量口径一）',
    'YD15': 'Target'
}, inplace=True)

df[['预测功率', 'Target']] = df[['预测功率', 'Target']] / 1000000

df.to_csv('./数据集/1.csv', index=False)