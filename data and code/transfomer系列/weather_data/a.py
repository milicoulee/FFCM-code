import pandas as pd

# 定义文件路径
weather_files = {
    'sunny': 'sunny.xlsx',
    'rainy': 'rainy.xlsx', 
    'cloudy': 'cloudy.xlsx'
}

# 遍历读取每个天气类型的文件并计算统计指标
for weather_type, file_path in weather_files.items():
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 检查Target列是否存在
        if 'Target' not in df.columns:
            print(f"Warning: 'Target' column not found in {file_path}")
            continue
            
        # 使用describe()函数计算统计指标
        print(f"\nStatistics for {weather_type.capitalize()} Days:")
        target_stats = df['Target'].describe(percentiles=[.25, .5, .75])
        
        # 添加偏度和峰度
        target_stats['skewness'] = df['Target'].skew()
        target_stats['kurtosis'] = df['Target'].kurt()
        
        # 保留4位小数
        target_stats = target_stats.round(4)
        
        print(target_stats.to_string())
        
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")