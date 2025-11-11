import os
new_working_directory = r"D:\毕业论文\傅立叶卷积_itransformer\傅立叶卷积_itransformer\transfomer系列"
os.chdir(new_working_directory)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models.iTransformer import  to_3d,to_4d,FourierUnit, Freq_Fusion,FFCM
import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def cal_eval(y_real, y_pred):
    
    

    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred, squared=True)
    rmse = mean_squared_error(y_real, y_pred, squared=False)  # RMSE and MAE are various on different scales
    mae = mean_absolute_error(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / (y_real + 1e-8))) * 100


    df_eval = pd.DataFrame({
        'R2': [round(r2, 4)],  
        'MSE': [round(mse, 6)], 
        'RMSE': [round(rmse, 6)], 
        'MAE': [round(mae, 4)]  
    }, index=['Eval'])
    
    return df_eval
# 1. 加载数据
# 假设 df 是您的 DataFrame，包含 ['windspeed_100m', 'windspeed_10m', 'windgusts_10m', 'Target']

# df = pd.read_csv('./数据集/Location1_hd.csv')
df = pd.read_csv('./spring_data.csv')
selected_features = ['windspeed_10', 'windspeed_30', 'windspeed_50', 'Target']
df = df[selected_features]
data = df.values
data_target = df['Target'] 
from sklearn.preprocessing import MinMaxScaler

# 2. 数据预处理 - 构造输入序列（25步）和输出序列（6步）
input_window = 9
output_window = 1
scaler = MinMaxScaler()
data = scaler.fit_transform(np.array(data))

def create_sequences(data, input_window, output_window):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        # 输入包含所有列，时间步为 input_window
        X.append(data[i:i + input_window, :])  # 输入的维度为 25 × 4
        # 输出只针对目标列，时间步为 output_window
        y.append(data[i + input_window:i + input_window + output_window, -1])  # 输出的维度为 6 × 1
    return np.array(X), np.array(y)

# 将数据转换为 NumPy 数组
data = df.values
X, y = create_sequences(data, input_window, output_window)

# 数据划分为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
test_size = 0.2  # 测试集比例
split_index = int(len(X) * (1 - test_size))  # 计算分割索引
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 3. 自定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. 定义 GRU 模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ffcm_module = FFCM(dim=4)
        # GRU 层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = to_4d(x, 5, 5)  # 转为四维
        # x = self.ffcm_module(x)  # 经过图卷积模块处理
        # x = to_3d(x)  # 转回三维
        # GRU 输出
        out, _ = self.gru(x)  # out: (batch_size, sequence_length, hidden_size)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc(out)  # 全连接层
        return out

# 初始化 GRU 模型
input_size = X_train.shape[2]  # 输入特征数
hidden_size = 64  # GRU 隐藏层大小
num_layers = 2    # GRU 层数
output_size = output_window  # 输出的时间步数
model = GRUModel(input_size, hidden_size, num_layers, output_size).to('cuda' if torch.cuda.is_available() else 'cpu')

# 5. 设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. 训练模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

num_epochs = 70
train_losses = []
saved_models = [] 
#%% 训练
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
    if epoch >= 1:
    # if epoch_loss <= 0.0049:
        model_filename = os.path.join('./保存最后10个/GRU/', f"GRU_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_filename)
        saved_models.append(model_filename)
        print(f"Model saved at epoch {epoch + 1} as {model_filename}")

#%% 加载单个模型
# 7. 保存模型
# save_path = "./save/GRU/8.pth"
# # torch.save(model.state_dict(), save_path)
# print(f"Model saved at {save_path}")


# loaded_model = GRUModel(input_size, hidden_size, num_layers, output_size).to('cuda' if torch.cuda.is_available() else 'cpu')
# loaded_model.load_state_dict(torch.load(save_path))
# loaded_model.eval()  # 切换为评估模式
# print(f"Model loaded from {save_path}")

# # 9. 预测
# loaded_model.eval()
# predictions, actuals = [], []
# with torch.no_grad():
#     for X_batch, y_batch in test_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)

#         # 预测
#         outputs = loaded_model(X_batch)
#         predictions.append(outputs.cpu().numpy())
#         actuals.append(y_batch.cpu().numpy())


# # 将结果转换为 NumPy 数组
# predictions = np.concatenate(predictions, axis=0)
# actuals = np.concatenate(actuals, axis=0)

# # 计算评价指标
# y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
# true_uninverse = scaler.inverse_transform(actuals)
# pred_uninverse = scaler.inverse_transform(predictions)


# df_eval = cal_eval(true_uninverse, pred_uninverse)  # 评估指标dataframe

# print(f"Evaluation result 整体:\n{df_eval}\n")

#%% 时间步指标
# columns_to_evaluate = list(range(-1, -output_window-1, -1))

# evaluation_results = []

# 使用 for 循环遍历要评估的列
# for col in columns_to_evaluate:
#     if col == -1:
#         # 如果列索引是 -1，取从倒数第一列到最后一列的所有数据
#         true_col = scaler.inverse_transform(actuals[:, col:])  # 反向变换 true
#         pred_col = scaler.inverse_transform(predictions[:, col:])  # 反向变换 pred
#     else:
#         # 否则，按常规方式取某一列
#         true_col = scaler.inverse_transform(actuals[:, col:col+1])  # 反向变换 true
#         pred_col = scaler.inverse_transform(predictions[:, col:col+1])  # 反向变换 pred

#     # 计算评估指标
#     df_eval = cal_eval(true_col, pred_col)

#     # 打印评估结果
#     print(f"Evaluation for column {col}:")
#     print(df_eval)
#     print("\n" + "="*50 + "\n")

#     # 将评估结果添加到列表中
#     evaluation_results.append(df_eval)

# df_pred_true = pd.DataFrame({
#     'Predict': pred_uninverse[-1000:, -2].flatten(),
#     'Real': true_uninverse[-1000:, -2].flatten()
# })

# # 绘制图表
# df_pred_true.plot(figsize=(12, 4))
# plt.title('Model Result')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.show()
#%%
r_squared_values = []
mse_values = []
mae_values =[] 
eval_results = [] 
a,b = 2,num_epochs +1
for epoch in range(a, b):  # 从epoch 21到30
    save_path = f"./保存最后10个/GRU/GRU_epoch_{epoch}.pth"  # 保存的模型路径
    loaded_model =GRUModel(input_size, hidden_size, num_layers, output_size).to('cuda' if torch.cuda.is_available() else 'cpu')   # 使用与训练时相同的模型架构
    loaded_model.load_state_dict(torch.load(save_path))  # 加载保存的参数
    loaded_model.to(device)  # 转移到设备
    loaded_model.eval()  # 切换为验证模式



    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = loaded_model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    # 将结果转换为 NumPy 数组
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    # 计算评价指标
    y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
    true_uninverse = scaler.inverse_transform(actuals)
    pred_uninverse = scaler.inverse_transform(predictions)

    df_eval = cal_eval(true_uninverse, pred_uninverse)  # 评估指标dataframe
    # 保存计算整体的 MSE
    # print(f"{save_path}:\n{df_eval}\n")
    eval_results.append({
    'epoch': epoch,
    'save_path': save_path,
    'eval_metrics': df_eval
    })
    r_squared = df_eval.get('R2', None)  # 获取'R2'列
    mse = df_eval.get('MSE', None)  # 获取'MSE'列
    mae = df_eval.get('MAE', None)  # 获取'MAE'列
    
    # 提取数值，假设是Series对象
    r_squared_values.append(r_squared.values[0])
    mse_values.append(mse.values[0])
    mae_values.append(mae.values[0])
    
r_squared_series3 = pd.Series(r_squared_values, index=range(a, b))
mse_series3 = pd.Series(mse_values, index=range(a, b))
mae_series3 = pd.Series(mae_values, index=range(a, b))
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
# plt.plot(r_squared_series.index, r_squared_series.values, marker='o', color='b', linestyle='-', markersize=6)
plt.plot(mse_series2.index, mse_series2.values, marker='o', color='b', linestyle='-', markersize=6)
plt.plot(mse_series3.index, mse_series3.values, marker='o',linestyle='-', markersize=6)
plt.title("R² values per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("R² Value", fontsize=12)

plt.grid(True)

plt.show()
