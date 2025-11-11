import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
new_working_directory = r"D:\毕业论文\傅立叶卷积_itransformer\傅立叶卷积_itransformer\transfomer系列"
os.chdir(new_working_directory)
# 假设 df 是你的数据集，包含了日期列 'date' 和其他特征
df = pd.read_excel('./数据集/风电2019.xlsx')  

# 将 'date' 列转换为 datetime 类型
df['date'] = pd.to_datetime(df['date'])

# 将数据按日期进行聚合，计算每日的均值
daily_data = df.resample('D', on='date').agg({
    # 'windspeed_10': 'mean',  # 每日风速均值
    # 'windspeed_30': 'mean',  # 每日风速均值
    'windspeed_50': 'mean',  # 每日风速均值
    'humidity': 'mean',       # 每日湿度均值
    # '气压(hPa)': 'mean',     # 每日气压均值
    'temperature': 'mean'        # 每日温度均值
}).reset_index()

# 选择用于聚类的特征
features = daily_data[[ 'humidity', 'temperature','windspeed_50']]	


# 标准化数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 使用 GMM 进行聚类
n_clusters = 3  # 假设将数据分为 3 类（晴天、阴天、雨天）
gmm = GaussianMixture(n_components=n_clusters)
gmm.fit(features_scaled)

# 获取每个样本的聚类标签
labels = gmm.predict(features_scaled)

# 将聚类标签添加到数据集
daily_data['weather_cluster'] = labels

# 查看聚类结果
# print(daily_data[['date',  '湿度(%)', '气压(hPa)', '温度(°)', 'weather_cluster']])

# 可视化聚类结果
plt.scatter(daily_data['temperature'], daily_data['humidity'], c=daily_data['weather_cluster'])
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Weather Classification (Sunny, Cloudy, Rainy) based on Clustering')
plt.show()
#%%
# 1. 提取各天气类型对应的日期
sunny_days = daily_data[daily_data['weather_cluster'] == 0]['date']
rainy_days = daily_data[daily_data['weather_cluster'] == 1]['date']
cloudy_days = daily_data[daily_data['weather_cluster'] == 2]['date']


# 生成日期过滤器（处理可能的时间戳格式）
sunny_dates = pd.to_datetime(sunny_days.dt.date).unique()
rainy_dates = pd.to_datetime(rainy_days.dt.date).unique()
cloudy_dates = pd.to_datetime(cloudy_days.dt.date).unique()

# 方法1：先计算每日均值，再计算天气类型整体均值（更稳健）
def calculate_daily_mean(target_df, date_list):
    """计算指定日期列表的日均值"""
    daily_means = []
    for date in date_list:
        # 筛选当日所有时间点数据
        day_data = target_df[target_df['date'].dt.date == date.date()]
        # 验证数据完整性（必须包含完整96个时间点）
        if len(day_data) == 96:
            daily_means.append(day_data['Target'].mean())
    return np.mean(daily_means) if daily_means else np.nan

# 方法2：直接计算所有时间点的全局均值（更敏感）
def calculate_global_mean(target_df, date_list):
    """计算指定日期列表的全时间点均值"""
    filtered_data = target_df[target_df['date'].dt.date.isin([d.date() for d in date_list])]
    return filtered_data['Target'].mean() if not filtered_data.empty else np.nan

# 计算结果（两种方法对比）
weather_stats = pd.DataFrame({
    'Weather Type': ['Sunny', 'Rainy', 'Cloudy'],
    'Daily Mean': [
        calculate_daily_mean(df, sunny_dates),
        calculate_daily_mean(df, rainy_dates),
        calculate_daily_mean(df, cloudy_dates)
    ],
    'Global Mean': [
        calculate_global_mean(df, sunny_dates),
        calculate_global_mean(df, rainy_dates),
        calculate_global_mean(df, cloudy_dates)
    ]
})

print("风速功率均值对比：")
print(weather_stats)


#%%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(daily_data['temperature'], 
                     daily_data['humidity'], 
                     daily_data['windspeed_50'], 
                     c=daily_data['weather_cluster'],
                     cmap='viridis',
                     s=50,
                     depthshade=True)

# 设置坐标轴标签和视角
ax.set_xlabel('Temperature (°C)', labelpad=14)
ax.set_ylabel('Humidity (%)', labelpad=14)
ax.set_zlabel('Wind Speed 50m (m/s)', labelpad=14)
ax.view_init(elev=25, azim=45)  # 调整观察角度

# 添加颜色条和图例
cbar = plt.colorbar(scatter, pad=0.15)
cbar.set_label('Weather Cluster')
plt.title('3D Weather Clustering (Temp-Humi-Wind)', y=1.02, fontsize=16)

# 添加网格增强可读性
ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "alpha":0.3})
ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "alpha":0.3})
ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "alpha":0.3})

plt.show()


#%%

from sklearn.decomposition import PCA

# 使用PCA降维到2D
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features_scaled)

# 创建带解释度的坐标轴标签
variance_ratio = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
sc = plt.scatter(principal_components[:,0], 
                 principal_components[:,1],
                 c=daily_data['weather_cluster'],
                 cmap='viridis',
                 alpha=0.8)

plt.xlabel(f'PC1 ({variance_ratio[0]*100:.1f}%)', fontsize=14)
plt.ylabel(f'PC2 ({variance_ratio[1]*100:.1f}%)', fontsize=14)
plt.title('PCA Projection of 3D Weather Features', pad=15,fontsize=16)
plt.colorbar(sc, label='Weather Cluster')

# 添加特征向量箭头
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
features = ['Temp', 'Humi', 'Wind']
for i, (comp1, comp2) in enumerate(loadings):
    plt.arrow(0, 0, comp1 * 3, comp2 * 3, 
              color='r', 
              width=0.05,
              head_width=0.3)
    plt.text(comp1 * 3.6, comp2 * 3.6, 
              features[i], 
              color='r',
              ha='center', 
              va='center')
plt.grid(alpha=0.2)
plt.show()


#%% 保存数据
# import pandas as pd
# import os
# from datetime import datetime

# # 创建保存目录
# save_dir = "./weather_data"
# os.makedirs(save_dir, exist_ok=True)

# # 合并天气标签到原始数据（带时间戳）
# def merge_weather_labels(df, daily_data):
#     # 创建日期键
#     df['date_day'] = df['date'].dt.normalize()
#     daily_data['date_day'] = daily_data['date'].dt.normalize()

#     # 合并标签
#     merged_df = pd.merge(
#         df,
#         daily_data[['date_day', 'weather_cluster']],
#         on='date_day',
#         how='left'
#     )
#     return merged_df.drop(columns=['date_day'])

# # 执行数据合并
# df_labeled = merge_weather_labels(df, daily_data)

# # 定义天气类型映射
# weather_map = {
#     2: 'sunny',
#     1: 'rainy',
#     0: 'cloudy'
# }

# # 生成带时间范围的保存文件名
# def generate_filename(weather_type, df):
#     min_date = df['date'].min().strftime('%Y%m%d')
#     max_date = df['date'].max().strftime('%Y%m%d')
#     return f"{weather_type}_data.xlsx"

# # 分割并保存数据
# for cluster_num, weather_name in weather_map.items():
#     # 筛选对应天气数据
#     weather_df = df_labeled[df_labeled['weather_cluster'] == cluster_num]

#     # 生成文件名
#     filename = generate_filename(weather_name, weather_df)
#     save_path = os.path.join(save_dir, filename)

#     # 保存数据（保留原始时间戳）
#     weather_df.to_excel(
#         save_path,
#         index=False,
#         sheet_name=weather_name.capitalize(),
#         engine='openpyxl'
#     )

#     # 打印保存信息
#     print(f"已保存 {weather_name} 数据：")
#     print(f"文件路径：{save_path}")
#     print(f"时间范围：{weather_df['date'].min()} 至 {weather_df['date'].max()}")
#     print(f"记录数量：{len(weather_df):,} 行")
#     print(f"文件大小：{os.path.getsize(save_path)/1024:.1f} KB\n")

# # 验证保存结果
# print(f"所有天气类型数据已保存至目录：{os.path.abspath(save_dir)}")

