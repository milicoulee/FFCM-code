import os
new_working_directory = r"D:\毕业论文\傅立叶卷积_itransformer\傅立叶卷积_itransformer\transfomer系列"
os.chdir(new_working_directory)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import random
# 设置随机种子
seed = 300
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据评估函数
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

# 数据预处理函数
def create_sequences(data, input_window, output_window, step_size=1):
    X, y = [], []
    for i in range(0, len(data) - input_window - output_window + 1, step_size):
        X.append(data[i:i + input_window, :])  # 输入的维度为 input_window × n_features
        y.append(data[i + input_window:i + input_window + output_window, :])  # 输出的维度为 output_window × n_features
    return np.array(X), np.array(y)

# 加载数据
df = pd.read_csv('./spring_data.csv')
selected_features = ['windspeed_10', 'windspeed_30', 'windspeed_50', 'Target']
df = df[selected_features]

# 数据标准化
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(df))
input_window = 9
output_window = 1

X, y = create_sequences(data_inverse, input_window, output_window)

# 数据划分
test_size = 0.2  # 测试集比例
split_index = int(len(X) * (1 - test_size))  # 计算分割索引
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 自定义数据集
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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# CNN-BiLSTM 模型
class CNN_BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN_BiLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 双向LSTM所以是hidden_size * 2

    def forward(self, x):
        # x 的形状: (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)  # 调整形状为 (batch_size, num_features, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # 调整形状回 (batch_size, sequence_length, num_features)
        x, _ = self.lstm(x)  # 双向LSTM
        x = self.fc(x[:, -1, :])  # 只取最后一个时间步的输出
        return x

# 初始化模型
model = CNN_BiLSTM(input_size=X_train.shape[2], hidden_size=64, output_size=output_window)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = model(X_batch)
        y_pred_list.append(y_pred)
        y_true_list.append(y_batch)

y_pred = torch.cat(y_pred_list, dim=0).numpy()
y_true = torch.cat(y_true_list, dim=0).numpy()

# 评估
eval_metrics = cal_eval(y_true, y_pred)
print(eval_metrics)

# 可视化预测结果
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
