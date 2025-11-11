import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
new_working_directory = r"D:\毕业论文\傅立叶卷积_itransformer\傅立叶卷积_itransformer\transfomer系列"
os.chdir(new_working_directory)
# 数据预处理
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models import iTransformer,iReformer,iFlashformer,iFlowformer,iInformer,Informer,model1,AND
from utils.timefeatures import time_features
# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 读取数据


#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('./数据集/风电2019.xlsx')

# 删除 'date' 列
df = df[df.columns.drop('date')]

# 计算所有变量之间的相关系数
correlation_matrix = df.corr()

# 打印相关系数矩阵
print(correlation_matrix)

# 绘制相关性热图
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt='.2f', 
            linewidths=0.5, annot_kws={'size': 15})  # 增加注释字体大小
# plt.title('特征相关性热图', fontsize=14)  # 增加标题字体大小
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()

