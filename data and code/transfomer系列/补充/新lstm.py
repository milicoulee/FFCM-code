import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models.iTransformer import to_3d, to_4d, FourierUnit, Freq_Fusion, FFCM
import random

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 计算评估指标
def cal_eval(y_real, y_pred):
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred, squared=True)
    rmse = mean_squared_error(y_real, y_pred, squared=False)  # RMSE and MAE are various on different scales
    mae = mean_absolute_error(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / (y_real + 1e-8))) * 100

    df_eval = pd.DataFrame({'R2': r2,
                            'MSE': mse, 'RMSE': rmse,
                            'MAE': mae, 'MAPE': mape},
                           index=['Eval'])
    df_eval = df_eval.round(6)
    return df_eval

# 1. 加载数据
df = pd.read_csv('./数据集/Location1_hd.csv')
df = df[df.columns.drop('date')]
selected_features = ['windspeed_100m', 'windspeed_10m', 'windgusts_10m', 'Target']
df = df[selected_features]

data = df.values
data_target = df['Target']

# 2. 数据预处理 - 构造输入序列（25步）和输出序列（6步）
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))
input_window = 25
output_window = 6
length_size = output_window

def create_sequences(data, input_window, output_window):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:i + input_window, :])  # 输入的维度为 25 × 4
        y.append(data[i + input_window:i + input_window + output_window, :])  # 输出所有特征，维度为 6 × 4
    return np.array(X), np.array(y)

# 将数据转换为 NumPy 数组
X, y = create_sequences(data_inverse, input_window, output_window)

# 数据划分为训练集和测试集
test_size = 0.2  # 测试集比例
split_index = int(len(X) * (1 - test_size))  # 计算分割索引
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. 构建 BiLSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)  # 输出维度为 4，预测4个特征

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出（历史滑动窗口的最终特征）
        out = self.fc(out)  # 全连接层，输出 4 个特征的预测值
        return out

# 初始化模型
input_size = X_train.shape[2]  # 输入特征数（列数）
hidden_size = 64  # LSTM 隐藏层大小
num_layers = 2    # LSTM 层数
output_size = 4   # 输出特征数（4个特征）

model = BiLSTM(input_size, hidden_size, num_layers, output_size).to('cuda' if torch.cuda.is_available() else 'cpu')

# 5. 设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 6. 训练模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

num_epochs = 30
train_losses = []

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

# 7. 测试模型
save_path = "./lstm.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")

# 8. 加载保存的模型
loaded_model = BiLSTM(input_size, hidden_size, num_layers, output_size).to(device)
loaded_model.load_state_dict(torch.load(save_path))
loaded_model.eval()  # 切换为评估模式
print(f"Model loaded from {save_path}")

# 9. 测试模型
loaded_model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 预测
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

df_eval = cal_eval(true_uninverse[:, -1], pred_uninverse[:, -1])  # 评估 `Target` 列
print(f"Evaluation result for Target column:\n{df_eval}")

# 绘制图表
df_pred_true = pd.DataFrame({
    'Predict': pred_uninverse[-1000:, -1].flatten(),  # 取 Target 列的预测结果
    'Real': true_uninverse[-1000:, -1].flatten()  # 取 Target 列的真实值
})

df_pred_true.plot(figsize=(12, 4))
plt.title('Model Result')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
