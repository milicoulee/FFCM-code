import pandas as pd
import matplotlib.pyplot as plt

# 文件路径字典（包含 FFCM 和 No_FFCM 的模型）
# 文件路径字典（包含 FFCM 和 No_FFCM 的模型）
files_ffcm = {
    "FFCM_iFlashformer": "./result1/时间步_FFCM_iFlashformer.csv",
    "FFCM_iFlowformer": "./result1/时间步_FFCM_iFlowformer.csv",
    "FFCM_iInformer": "./result1/时间步_FFCM_iInformer.csv",
    "FFCM_iReformer": "./result1/时间步_FFCM_iReformer.csv",
    "FFCM_iTransformer": "./result1/时间步_FFCM_iTransformer.csv",
}

files_no_ffcm = {
    "iFlashformer": "./result1/时间步_iFlashformer.csv",
    "iFlowformer": "./result1/时间步_iFlowformer.csv",
    "iInformer": "./result1/草稿.csv",
    "iReformer": "./result1/时间步_iReformer.csv",
    "iTransformer": "./result1/时间步_iTransformer.csv",
}

# 定义颜色：相同基模型使用相同颜色
colors = {
    "iFlashformer": "b",
    "iFlowformer": "g",
    "iInformer": "r",
    "iReformer": "c",
    "iTransformer": "m"
}

# 要比较的指标
metrics = ["R2", "MSE", "RMSE", "MAE"]

# 初始化绘图
fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # 2x2 布局
fig.suptitle("Comparison of Evaluation Metrics With and Without FFCM", fontsize=18)

# 绘制每个指标的对比
for idx, metric in enumerate(metrics):
    row, col = divmod(idx, 2)  # 确定子图位置
    ax = axs[row, col]
    
    # 读取每对模型的文件并绘制
    for (model_ffcm, file_ffcm), (model_no_ffcm, file_no_ffcm) in zip(files_ffcm.items(), files_no_ffcm.items()):
        # 提取基模型名字
        base_model = model_ffcm.split('_')[-1]  # 获取最后一部分，如 "iInformer"
        print(f"Processing Base Model: {base_model}")
        
        # 确保颜色字典的键与提取的模型名称一致
        if base_model not in colors:
            print(f"Error: {base_model} not found in colors dictionary.")
            continue
        
        # 读取 CSV 文件
        try:
            print(f"Reading file: {file_ffcm} and {file_no_ffcm}")
            df_ffcm = pd.read_csv(file_ffcm)
            df_no_ffcm = pd.read_csv(file_no_ffcm)
            
            # 绘制 FFCM 和 No_FFCM 的指标对比
            ax.plot(df_ffcm['Steps'], df_ffcm[metric], marker='o', linestyle='-', color=colors[base_model], label=f"FFCM {base_model}")
            ax.plot(df_no_ffcm['Steps'], df_no_ffcm[metric], marker='x', linestyle='--', color=colors[base_model], label=f"{base_model}")
        except Exception as e:
            print(f"Error plotting for {base_model}: {e}")
            continue
    
    # 设置子图属性
    ax.set_title(f"Comparison of {metric}", fontsize=14)
    ax.set_xlabel("Prediction Steps", fontsize=12)
    ax.set_ylabel(f"{metric} Value", fontsize=12)
    ax.set_xticks([1, 2, 3, 4, 5, 6])  # 假设横轴步数是 1 到 6
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6'])  # 横轴刻度标签
    ax.legend(fontsize=9, loc='best')  # 图例放置在最佳位置
    ax.grid(True)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为标题留出空间
plt.show()
