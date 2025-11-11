import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



# 设置工作目录
new_working_directory = r"D:\毕业论文\傅立叶卷积_itransformer\傅立叶卷积_itransformer\transfomer系列"
os.chdir(new_working_directory)

# 1. 读取数据
file_path = './数据集/风电2019.xlsx'  # 数据文件路径
data = pd.read_excel(file_path)  # 确保是 Excel 文件

# 2. 去掉非数值列（如日期）
numeric_data = data.select_dtypes(include=['float64', 'int64'])
if numeric_data.empty:
    raise ValueError("输入数据中没有数值列，请检查数据格式！")

# 3. 计算相关性矩阵
correlation_matrix = numeric_data.corr()

# 4. 绘制相关性热力图，调整字体大小
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
    annot_kws={"size": 10},  # 调整相关系数数字字体大小
    xticklabels=correlation_matrix.columns,
    yticklabels=correlation_matrix.columns
)
plt.xticks(fontsize=10, rotation=90)  # 调整变量名称字体大小和旋转角度
plt.yticks(fontsize=10, rotation=0)   # 调整y轴变量字体大小
plt.tight_layout()

# 保存热力图
plt.savefig('correlation_heatmap_larger_labels.png')
plt.show()
