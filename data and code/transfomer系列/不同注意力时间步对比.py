import pandas as pd
import matplotlib.pyplot as plt

# 文件路径（包含 ./result 目录）
files = {
    "FFCM_iFlashformer": "./result1/时间步_FFCM_iFlashformer.csv",
    "FFCM_iFlowformer": "./result1/时间步_FFCM_iFlowformer.csv",
    "FFCM_iInformer": "./result1/时间步_FFCM_iInformer.csv",
    "FFCM_iReformer": "./result1/时间步_FFCM_iReformer.csv",
    "FFCM_iTransformer": "./result1/时间步_FFCM_iTransformer.csv"
}

# 指标和颜色设置
metrics = ['R2', 'MSE', 'RMSE', 'MAE']  # 指标名称
colors = ['b', 'g', 'r', 'c', 'm']  # 不同模型的颜色
models = list(files.keys())  # 模型名称

# 读取所有文件的数据
data = {}
for model, file_path in files.items():
    data[model] = pd.read_csv(file_path)

# 绘图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2x2 布局
fig.suptitle("Comparison of Evaluation Metrics Across Models (FFCM)", fontsize=16)

for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]  # 获取子图
    for model, color in zip(models, colors):
        df = data[model]
        ax.plot(df['Steps'], df[metric], marker='o', color=color, label=model)
    ax.set_title(metric, fontsize=12)
    ax.set_xlabel("Prediction Steps", fontsize=10)
    ax.set_ylabel(f"{metric} Value", fontsize=10)
    ax.set_xticks([1, 2, 3, 4, 5, 6])  # 修改横轴刻度为 1 到 6
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6'])  # 设置刻度标签
    ax.legend(fontsize=9, loc='best')  # 图例放置在最佳位置
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局
plt.show()
