import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 文件路径字典（模型及其对应 CSV 文件路径）
files = {
    "FFCM_iFlashformer": "./result1/整体_FFCM_iFlashformer.csv",
    "FFCM_iFlowformer": "./result1/整体_FFCM_iFlowformer.csv",
    "FFCM_iInformer": "./result1/整体_FFCM_iInformer.csv",
    "FFCM_iReformer": "./result1/整体_FFCM_iReformer.csv",
    "FFCM_iTransformer": "./result1/整体_FFCM_iTransformer.csv"
}

# 初始化存储各模型指标的数据字典
metrics_data = {
    "Model": [],
    "MSE": [],
    "MAE": []
}

# 遍历文件，读取指标
for model, file_path in files.items():
    try:
        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        
        # 假设 CSV 文件中包含 MSE, MAE 列
        mse = df["MSE"].iloc[0]  # 提取 MSE 指标
        mae = df["MAE"].iloc[0]  # 提取 MAE 指标
        
        # 添加到数据字典
        metrics_data["Model"].append(model)
        metrics_data["MSE"].append(mse)
        metrics_data["MAE"].append(mae)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# 转换为 DataFrame
metrics_df = pd.DataFrame(metrics_data)

# ==============================
# 绘制双坐标轴柱状图（MSE 和 MAE）
# ==============================
x = np.arange(len(metrics_df["Model"]))  # 模型位置
width = 0.4  # 每组柱子的宽度

fig, ax1 = plt.subplots(figsize=(12, 6))

# 绘制 MSE 柱状图（主坐标轴）
bar1 = ax1.bar(x - width / 2, metrics_df["MSE"], width, label="MSE", color="blue")
ax1.set_xlabel("Models", fontsize=12)
ax1.set_ylabel("MSE", fontsize=12, color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.set_xticks(x)
ax1.set_xticklabels(metrics_df["Model"], rotation=15, fontsize=10)
ax1.grid(axis="y", linestyle="--", alpha=0.7)

# 绘制 MAE 柱状图（次坐标轴）
ax2 = ax1.twinx()  # 创建共享 X 轴的次坐标轴
bar2 = ax2.bar(x + width / 2, metrics_df["MAE"], width, label="MAE", color="orange")
ax2.set_ylabel("MAE", fontsize=12, color="orange")
ax2.tick_params(axis='y', labelcolor="orange")

# 添加标题
plt.title("Comparison of MSE and MAE Across Models", fontsize=16)

# 显示具体数值
for bar in bar1:
    height = bar.get_height()
    ax1.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 偏移值
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

for bar in bar2:
    height = bar.get_height()
    ax2.annotate(f'{height:.4f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 偏移值
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

# 调整布局并显示图例
fig.tight_layout()
plt.show()
