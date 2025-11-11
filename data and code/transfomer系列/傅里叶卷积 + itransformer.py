#%% 调包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

plt.rc('font',family='Arial')
plt.style.use("ggplot")
# 自己写的函数文件functionfile.py
# 如果需要调整TSlib-test.ipynb文件的路径位置 注意同时调整导入的路径
from models import iTransformer,iReformer,iFlashformer,iFlowformer,iInformer,Informer,model1,Transformer
from utils.timefeatures import time_features
# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import torch
import numpy as np
import random

#%%
def plot1(train_loss, lrs):
    """可视化训练损失和学习率变化。"""
    epochs = range(1, len(train_loss) + 1)

    # 创建损失和学习率的双轴图
    fig, ax1 = plt.subplots()

    # 训练损失曲线
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='tab:blue')
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 学习率曲线
    ax2 = ax1.twinx()  # 共享x轴
    ax2.set_ylabel('Learning Rate', color='tab:orange')
    ax2.plot(epochs, lrs, label='Learning Rate', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # 图例
    fig.tight_layout()  # 调整布局
    plt.title('Train Loss and Learning Rate Over Epochs')
    plt.show()
    
def plot2(train_loss, val_loss, lrs):
    epochs = range(1, len(train_loss) + 1)

    # 创建损失和学习率的双轴图
    fig, ax1 = plt.subplots()

    # 训练损失和验证损失曲线
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:blue', linestyle='-')
    ax1.plot(epochs, val_loss, label='Validation Loss', color='tab:green', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 学习率曲线
    ax2 = ax1.twinx()  # 共享x轴
    ax2.set_ylabel('Learning Rate', color='tab:orange')
    ax2.plot(epochs, lrs, label='Learning Rate', color='tab:orange', linestyle=':')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # 图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    fig.tight_layout()  # 调整布局
    plt.title('Train Loss, Validation Loss, and Learning Rate Over Epochs')
    plt.show()


def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    """
    数据加载器函数，用于加载和预处理时间序列数据，以用于训练模型。

    仅仅适用于 多变量预测多变量（可以单独取单变量的输出），或者单变量预测单变量
    也就是y里也会有外生变量？？

    参数:
    - window: 窗口大小，用于截取输入序列的长度。
    - length_size: 目标序列的长度。
    - batch_size: 批量大小，决定每个训练批次包含的数据样本数量。
    - data: 输入时间序列数据。
    - data_mark: 输入时间序列的数据标记，用于辅助模型训练或增加模型的多样性。

    返回值:
    - dataloader: 数据加载器，用于批量加载处理后的训练数据。
    - x_temp: 处理后的输入数据。
    - y_temp: 处理后的目标数据。
    - x_temp_mark: 处理后的输入数据的标记。
    - y_temp_mark: 处理后的目标数据的标记。
    """

    # 构建模型的输入
    seq_len = window
    sequence_length = seq_len + length_size
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])


    # 划分x与y
    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    # 转换为Tensor和数据类型
    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)
    # print(x_temp.shape)
    # print(y_temp.shape)
    # print(x_temp_mark.shape)
    # print(y_temp_mark.shape)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)


    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark


def model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device,scheduler=None, print_train=False,model_save_dir='./保存最后10个/'):
    """
    训练模型并应用早停机制。
    参数:
        net (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        length_size (int): 输出序列的长度。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (torch.nn.Module): 损失函数。
        num_epochs (int): 总训练轮数。
        device (torch.device): 设备（CPU或GPU）。
        print_train (bool, optional): 是否在训练中打印进度，默认为False。
    返回:
        net (torch.nn.Module): 训练好的模型。
        train_loss (list): 训练过程中每个epoch的平均训练损失列表。
        best_epoch (int): 达到最佳验证损失的epoch数。
    """

    train_loss = []  # 用于记录每个epoch的平均训练损失
    print_frequency = 10  # 计算打印训练状态的频率
    lrs = [] 
    saved_models = [] 
    print('无验证集')

    for epoch in range(num_epochs):
        total_train_loss = 0  # 初始化一个epoch的总损失

        net.train()  # 将模型设置为训练模式
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(
                device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  # 清空梯度
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()  # 前向传播
            # print(preds.shape)
            # preds = preds[:,-1]#单步预测
            # preds = preds[:,:,-1]#多步预测
            # preds = preds[:,:]
            
            # print(preds.shape)
            # labels = labels[:, -length_size:,-1].squeeze()
            labels = labels[:, -length_size:,:].squeeze()
            loss = criterion(preds, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss.item()  # 累加损失值

        avg_train_loss = total_train_loss / len(train_loader)  # 计算该epoch的平均损失
        train_loss.append(avg_train_loss)  # 将平均损失添加到列表中
        
        # 更新学习率
        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        # print(f"Epoch {epoch + 1},  Learning Rate: {current_lr:.8f}")
        
        # 如果设置为打印训练状态，则按频率打印
        if print_train:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}")
                
        # if epoch >= num_epochs - 30:
        # if avg_train_loss <= 0.0047:
            
        model_filename = os.path.join(model_save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(net.state_dict(), model_filename)
        saved_models.append(model_filename)
        # print(f"Model saved at epoch {epoch + 1} as {model_filename}")
    return net, train_loss, epoch + 1,lrs
def model_train_val(net, train_loader, val_loader, length_size, optimizer, criterion, scheduler, num_epochs, device, early_patience=0.15, print_train=False):
    """
    训练模型并应用早停机制。

    参数:
        model (torch.nn.Module): 待训练的模型。
        train_loader (torch.utils.data.DataLoader): 训练数据加载器。
        val_loader (torch.utils.data.DataLoader): 验证数据加载器。
        optimizer (torch.optim.Optimizer): 优化器。
        criterion (torch.nn.Module): 损失函数。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        num_epochs (int): 总训练轮数。
        device (torch.device): 设备（CPU或GPU）。
        early_patience (float, optional): 早停耐心值，默认为0.15 * num_epochs。
        print_train: 是否打印训练信息。
    返回:
        torch.nn.Module: 训练好的模型。
        list: 训练过程中每个epoch的平均训练损失列表。
        list: 训练过程中每个epoch的平均验证损失列表。
        int: 早停触发时的epoch数。
    """

    train_loss = []  # 用于记录每个epoch的平均损失
    val_loss = []  # 用于记录验证集上的损失，用于早停判断
    print_frequency = num_epochs / 10  # 计算打印频率
    lrs = []
    early_patience_epochs = int(early_patience * num_epochs)  # 早停耐心值（转换为epoch数）
    best_val_loss = float('inf')  # 初始化最佳验证集损失
    early_stop_counter = 0  # 早停计数器
    print('有验证集')

    for epoch in range(num_epochs):
        total_train_loss = 0  # 初始化一个epoch的总损失

        net.train()  # 将模型设置为训练模式
        for i, (datapoints, labels, datapoints_mark, labels_mark) in enumerate(train_loader):
            datapoints, labels, datapoints_mark, labels_mark = datapoints.to(device), labels.to(device), datapoints_mark.to(device), labels_mark.to(device)
            optimizer.zero_grad()  # 清空梯度
            preds = net(datapoints, datapoints_mark, labels, labels_mark, None).squeeze()  # 前向传播
            # preds = preds[:, -1]#单步预测
            # labels = labels[:, -length_size:].squeeze()  # 注意这一步
            
            preds = preds[:, :, -1]#多步预测
            labels = labels[:, -length_size: , -1].squeeze() 
            # print(labels.shape)
            loss = criterion(preds, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            total_train_loss += loss.item()  # 累加损失值

        avg_train_loss = total_train_loss / len(train_loader)  # 计算本epoch的平均损失
        train_loss.append(avg_train_loss)  # 记录平均损失

        with torch.no_grad():  # 关闭自动求导以节省内存和提高效率
            total_val_loss = 0
            for val_x, val_y, val_x_mark, val_y_mark in val_loader:
                val_x, val_y, val_x_mark, val_y_mark = val_x.to(device), val_y.to(device), val_x_mark.to(device), val_y_mark.to(device)  # 将数据移到GPU
                pred_val_y = net(val_x, val_x_mark, val_y, val_y_mark, None).squeeze()  # 前向传播
                # pred_val_y =pred_val_y[:,:,-1]
                # val_y = val_y[:, -length_size:].squeeze()  # 注意这一步
                pred_val_y =pred_val_y[:, :, -1]
                val_y = val_y[:, -length_size:, -1].squeeze()  # #########################
                val_loss_batch = criterion(pred_val_y, val_y)  # 计算损失
                total_val_loss += val_loss_batch.item()

            avg_val_loss = total_val_loss / len(val_loader)  # 计算本epoch的平均验证损失
            val_loss.append(avg_val_loss)  # 记录平均验证损失

            # scheduler.step(avg_val_loss)  # 更新学习率（基于当前验证损失）
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            #打印
            print(f"Epoch {epoch + 1},  Learning Rate: {current_lr:.6f}")

        
        # 打印训练信息
        if print_train == True:
            if (epoch + 1) % print_frequency == 0:
                print(f"Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # 早停判断
        save_path = "best_model.pth" 
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0  # 重置早停计数器
            torch.save(net.state_dict(), save_path)
            print(f"Val loss improved  {avg_val_loss:.6f}. Model saved to {save_path}.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_patience_epochs:
                print(f'Early stopping triggered at epoch {epoch + 1}.')
                break  # 早停
    net.train()  # 恢复训练模式
    return net, train_loss, val_loss, epoch + 1,lrs


# 计算点预测的评估指标
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
        'MAE': [round(mae, 4)]  ,
        'MAPE': [round(mape, 4)] 
    }, index=['Eval'])
    
    return df_eval

#%%  取数据
# df = pd.read_csv('./数据集/Location1_hd.csv')
# df = pd.read_csv('./data/feng2019.csv')
# df = pd.read_csv('./data/test_data_多变量.csv')
df = pd.read_excel('./数据集/风电2019.xlsx')  
# df = pd.read_excel('./数据集/光伏2019.xlsx')  
# df = pd.read_csv('./autumn_data.csv')
# df = pd.read_excel('./weather_data/rainy.xlsx') 
# df = pd.read_csv('./数据集/1.csv')
# df = df.iloc[18000: , :]
# df = pd.read_csv('./数据集/Task1_W_Zone1.csv')
# df = pd.read_csv('./数据集/data1.csv')
# df = df.iloc[8738: , :]

# df = pd.read_excel('./数据集/内蒙古.xlsx')
# df = df.iloc[:35584,:]

data_target = df['Target']  # 预测的目标变量
data_dim = 4
# data = df[['windspeed_100m', 'windspeed_10m', 'windgusts_10m', 'Target']]
# data = df[['风速', '预测功率', '实际风速', 'Target']]
df['date'] = pd.to_datetime(df['date'])

# data = df[['windspeed_10', 'windspeed_30', 'windspeed_50', 'Target']]

# data = df[[1, 2, 3, 'Target']]


# data_dim = df[df.columns.drop('date')].shape[1]  # 一共多少个变量
# data = df[df.columns.drop('date')]  # 选取所有的数据


#%% 季节拆分
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['season'] = df['date'].dt.month.apply(get_season)

# 不限制年份，提取所有数据并按季节分割
spring_data = df[df['season'] == 'Spring']
summer_data = df[df['season'] == 'Summer']
autumn_data = df[df['season'] == 'Autumn']
winter_data = df[df['season'] == 'Winter']
# # 春季数据为 CSV 文件
# spring_data.to_csv('spring_data.csv', index=False)

# # 夏季数据为 CSV 文件
# summer_data.to_csv('summer_data.csv', index=False)

# # 秋季数据为 CSV 文件
# autumn_data.to_csv('autumn_data.csv', index=False)

# # 冬季数据为 CSV 文件
# winter_data.to_csv('winter_data.csv', index=False)

# 打印各个季节的数据长度
print(f"Spring data length: {len(spring_data)}")
print(f"Summer data length: {len(summer_data)}")
print(f"Autumn data length: {len(autumn_data)}")
print(f"Winter data length: {len(winter_data)}")


data =  spring_data[['windspeed_10', 'windspeed_30', 'windspeed_50', 'Target']]
# data = summer_data[['windspeed_100m', 'windspeed_10m', 'windgusts_10m', 'Target']] # 4574
# data = autumn_data[['windspeed_100m', 'windspeed_10m', 'windgusts_10m', 'Target']]
# data = winter_data[['windspeed_100m', 'windspeed_10m', 'windgusts_10m', 'Target']]



#%% 异常值检测 
# impor本的的本文模型t pandas as pd
# import numpy as np
# import pandas as pd

# # 假设 df 已经是你的 DataFrame
# df['date'] = pd.to_datetime(df['date'])
# df = df.sort_values(by='date')

# # 假设需要对某列（例如 'WINDSPEED'）进行异常值处理
# column_name = 'Target'

# # 计算 IQR
# Q1 = df[column_name].quantile(0.25)
# Q3 = df[column_name].quantile(0.75)
# IQR = Q3 - Q1

# # 计算异常值的上下界
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # 找出异常值
# outliers = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)

# # 填充异常值：使用前后值的插值
# df[column_name] = df[column_name].where(~outliers, np.nan)  # 将异常值设为NaN

# # 使用线性插值填充缺失值（异常值）
# df[column_name] = df[column_name].interpolate(method='linear', limit_direction='both')

# # 也可以选择使用均值填充：替换 NaN 为该列的均值
# # df[column_name] = df[column_name].fillna(df[column_name].mean())

# # 输出处理后的数据
# print(df.head())



#%% 划分数据

# 时间戳
df_stamp = df[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp = time_features(df_stamp, timeenc=1, freq='T')  # 这一步很关键，注意数据的freq

# 无验证集

# 数据归一化
scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))

data_length = len(data_inverse)
train_set = 0.8

data_train = data_inverse[:int(train_set * data_length), :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
data_train_mark = data_stamp[:int(train_set * data_length), :]
data_test = data_inverse[int(train_set * data_length):, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
data_test_mark = data_stamp[int(train_set * data_length):, :]
# split_index = -96*7

# data_train = data_inverse[:split_index, :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
# data_train_mark = data_stamp[:split_index, :]
# data_test = data_inverse[split_index:, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
# data_test_mark = data_stamp[split_index:, :]
# n_feature = data_dim

window = 9  # 模型输入序列长度 这里只能是 12345 的这种次方倍数 这个改的话 model层的209行也得改
length_size = 1  # 预测结果的序列长度
batch_size = 64

train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size, data_train, data_train_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test, data_test_mark)


#%% 模型参数                                                                       
from math import sqrt
num_epochs = 100   # 训练迭代次数 300论以后趋于稳定 0.001+500最佳 这个别改 loss维持再0.0010徘徊
learning_rate = 0.001 # 学习率
early_patience = 0.15  # 训练迭代的早停比例 即patience=0.25*num_epochs
# scheduler_patience = 2 #学习率调整的patience
# scheduler_patience = int(0.25 * num_epochs)  # 转换为整数  

class Config:
    def __init__(self):
        # basic
        self.seq_len = window  # input sequence length
        self.label_len = int(window / 2)  # start token length
        self.pred_len = length_size  # 预测序列长度
        self.freq = 't'  # 时间的频率，
        # 模型训练
        self.batch_size = batch_size  # 批次大小
        self.num_epochs = num_epochs  # 训练的轮数
        self.learning_rate = learning_rate  # 学习率
        self.stop_ratio = early_patience  # 早停的比例
        # 模型 define
        self.dec_in = data_dim  # 解码器输入特征数量, 输入几个变量就是几
        self.enc_in = data_dim  # 编码器输入特征数量
        self.c_out = 1  # 输出维度##########这个很重要
        # 模型超参数
        self.d_model = 64  # 模型维度
        self.n_heads = 8  # 多头注意力头数
        self.dropout = 0.1  # 丢弃率
        self.e_layers = 2  # 编码器块的数量
        self.d_layers = 1  # 解码器块的数量
        self.d_ff = 64  # 全连接网络维度
        self.factor = 5  # 注意力因子
        self.activation = 'gelu'  # 激活函数
        self.channel_independence = 0  # 频道独立性，0:频道依赖，1:频道独立

        self.top_k = 5  # TimesBlock中的参数
        self.num_kernels = 6  # Inception中的参数
        self.distil = 1  # 是否使用蒸馏，1为True
        # 一般不需要动的参数
        self.embed = 'timeF'  # 时间特征编码方式
        self.output_attention = 0  # 是否输出注意力
        self.task_name = 'short_term_forecast'  # 模型的任务，一般不动但是必须这个参数
        self.moving_avg = window - 1  # Autoformer中的参数
          
        self.FFCM = 1 #1开启0关闭

#%%
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eval_results = []  # 存储每次评估结果
training_times = [] 


config = Config()

start_time = time.time()

# model_type = 'DiffAttn-iTransformer'
model_type = '完整模型'
net = iTransformer.Model(config).to(device)
# model_type = 'FFCM_iReformer'
# net = iReformer.Model(config).to(device)
# model_type = 'FFCM_iFlashformer'
# net = iFlashformer.Model(config).to(device)
# model_type = 'FFCM_iFlowformer'
# net = iFlowformer.Model(config).to(device)
# model_type = 'FFCM_iInformer'
# net = iInformer.Model(config).to(device)

# model_type = 'FFCM_改进差分注意力'
# net = model1.Model(config).to(device)
print(f'模型：{model_type}')

criterion = nn.MSELoss().to(device)  # 损失函数
optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 优化器

# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=scheduler_patience, verbose=True)  # 学习率调整策略
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)



trained_model, train_loss, final_epoch,lrs = model_train(net, train_loader, length_size, optimizer, criterion, num_epochs, device, print_train=True
                                                          ,scheduler= None )
torch.save(net.state_dict(), 'trained_model_state.pth')


# save_path = "trained_model_state.pth"    
save_path = f"./保存最后10个/model_epoch_{num_epochs}.pth"

loaded_model = net  # 使用与训练时相同的模型架构
loaded_model.load_state_dict(torch.load(save_path))  # 加载保存的参数
loaded_model.to(device)  # 转移到设备
loaded_model.eval()  # 切换为验证模式



# plot1(train_loss, lrs)
# # plot2(train_loss,val_loss, lrs)

# trained_model.eval()  # 模型转换为验证模式

#结束时间
end_time = time.time()
training_time = end_time - start_time
training_times.append(training_time)

# 预测并调整维度
pred = loaded_model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
# pred = trained_model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))

true = y_test[:, -length_size:, -1:].detach().cpu()
pred = pred.detach().cpu()
# 检查pred和true的维度并调整
print("Shape of true before adjustment:", true.shape)
print("Shape of pred before adjustment:", pred.shape)

# 可能需要调整pred和true的维度，使其变为二维数组
true = true[:, :, -1]
pred = pred[:, :, -1]  # 假设需要将pred调整为二维数组，去掉最后一维
# true = np.array(true)
# 假设需要将true调整为二维数组

print("Shape of pred after adjustment:", pred.shape)
print("Shape of true after adjustment:", true.shape)

# 这段代码是为了重新更新scaler，因为之前定义的scaler是十六维，这里重新根据目标数据定义一下scaler
y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
# pred_uninverse = scaler.inverse_transform(pred[:, -1:])    # 如果是多步预测，选取最后一列
# true_uninverse = scaler.inverse_transform(true[:, -1:])
# true, pred = true_uninverse, pred_uninverse

# 反归一化预测值和真实值
pred_uninverse = scaler.inverse_transform(pred)  # 如果是多步预测，选取最后一列
true_uninverse = scaler.inverse_transform(true)
# %% 保存计算整体的 MSE
df_eval = cal_eval(true_uninverse, pred_uninverse)  # 评估指标dataframe
df_eval_filename = f'./result2/整体_{model_type}.csv'.replace("+", "_") #保存
df_eval.to_csv(df_eval_filename, index=False)   
print(df_eval)

eval_results.append(df_eval)  # 存储本次评估结果




#%%每步的MSE

# columns_to_evaluate = list(range(-1, -length_size-1, -1))
# evaluation_results = []
# for col in columns_to_evaluate:
#     if col == -1:
#         # 如果列索引是 -1，取从倒数第一列到最后一列的所有数据
#         true_col = scaler.inverse_transform(true[:, col:])  # 反向变换 true
#         pred_col = scaler.inverse_transform(pred[:, col:])  # 反向变换 pred
#     else:

#         true_col = scaler.inverse_transform(true[:, col:col+1])  # 反向变换 true
#         pred_col = scaler.inverse_transform(pred[:, col:col+1])  # 反向变换 pred

#     # 计算评估指标
#     df_eval = cal_eval(true_col, pred_col)

#     print(f"Evaluation for column {col}:")
#     print(df_eval)
#     print("\n" + "="*50 + "\n")
#     evaluation_results.append(df_eval)

# all_results_df = pd.concat(evaluation_results, axis=0)
# steps = list(range(length_size, 0, -1))  # 预测步长 [length_size, ..., 3, 2, 1]
# all_results_df['Steps'] = steps

# all_results_filename = f'./result2/时间步_{model_type}.csv'.replace("+", "_") #保存


# all_results_df.to_csv(all_results_filename, index=False)


#%%
r_squared_values = [] 
rmse_values = []
mae_values =[]
a,b = 1,num_epochs+1
# a,b = 38,39
for epoch in range(a, b):  # 从epoch 21到30
    # save_path = "trained_model_state.pth"  # 保存的模型路径
    save_path = f"./保存最后10个/model_epoch_{epoch}.pth"  # 保存的模型路径
    loaded_model = net  # 使用与训练时相同的模型架构
    loaded_model.load_state_dict(torch.load(save_path))  # 加载保存的参数
    loaded_model.to(device)  # 转移到设备
    loaded_model.eval()  # 切换为验证模式

    # 预测
    pred = loaded_model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))


    true = y_test[:, -length_size:, -1:].detach().cpu()
    pred = pred.detach().cpu()


    # 可能需要调整pred和true的维度，使其变为二维数组
    true = true[:, :, -1]
    pred = pred[:, :, -1]  # 假设需要将pred调整为二维数组，去掉最后一维
    # print("Shape of pred after adjustment:", pred.shape)
    # print("Shape of true after adjustment:", true.shape)

    y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
    # pred_uninverse = scaler.inverse_transform(pred[:, -1:])    # 如果是多步预测，选取最后一列
    # true_uninverse = scaler.inverse_transform(true[:, -1:])
    # true, pred = true_uninverse, pred_uninverse

    # 反归一化预测值和真实值
    pred_uninverse = scaler.inverse_transform(pred)  # 如果是多步预测，选取最后一列
    true_uninverse = scaler.inverse_transform(true)
    #  保存计算整体的 MSE
    df_eval = cal_eval(true_uninverse, pred_uninverse)  # 评估指标dataframe
    df_eval_filename = f'./result2/整体_{model_type}.csv'.replace("+", "_") #保存
    df_eval.to_csv(df_eval_filename, index=False)   
    # print(f"{save_path}:\n{df_eval}\n")
    eval_results.append({
    'epoch': epoch,
    'save_path': save_path,
    'eval_metrics': df_eval
    })
    # 从DataFrame中获取R²、MSE和MAE的值
    r_squared = df_eval.get('R2', None)  # 获取'R2'列
    rmse = df_eval.get('RMSE', None)  # 获取'MSE'列
    mae = df_eval.get('MAE', None)  # 获取'MAE'列
    
    # 提取数值，假设是Series对象
    r_squared_values.append(r_squared.values[0])
    rmse_values.append(rmse.values[0])
    mae_values.append(mae.values[0])
    
import matplotlib.pyplot as plt
import pandas as pd
r_squared_series = pd.Series(r_squared_values, index=range(a, b))
rmse_series = pd.Series(rmse_values, index=range(a, b))
mae_series = pd.Series(mae_values, index=range(a, b))

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制 R²、MSE 和 MAE 的折线图
# plt.plot(r_squared_series.index, r_squared_series.values, marker='o', color='b', linestyle='-', label="R²", markersize=6)
plt.plot(rmse_series.iloc[50:].index, rmse_series.iloc[50:].values, marker='s', color='g', linestyle='-', label="MSE", markersize=6)
# plt.plot(mae_series.index, mae_series.values, marker='^', color='r', linestyle='-', label="MAE", markersize=6)

# 设置图表标题和标签
plt.title("R², MSE, and MAE per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Value", fontsize=12)

# 显示网格
plt.grid(True)

# 添加图例
plt.legend()

# 显示图表
plt.show()

print(f'模型：{model_type}')
# 创建一个函数来提取和打印所有相关指标


# 创建一个函数来提取和打印所有相关指标
def print_metrics(index, r_squared_series, mse_series, mae_series):
    r_squared_value = r_squared_series[index]
    mse_value = mse_series[index]
    mae_value = mae_series[index]

    # 使用 pandas DataFrame 来格式化输出
    eval_df = pd.DataFrame({
        'R2': [r_squared_value],
        'RMSE': [mse_value],
        'MAE': [mae_value]
    }, index=[f"Eval {index}"])

    print(eval_df.to_string(header=True, index=True, float_format='%.4f'))

# 获取 R 方最大值的轮次和数值
max_index = r_squared_series.idxmax()
max_value = r_squared_series.max()

# 获取最小 MSE 和 MAE 的轮次和数值
min_rmse_index = rmse_series.idxmin()
min_rmse_value = rmse_series.min()

min_mae_index = mae_series.idxmin()
min_mae_value = mae_series.min()

# 输出 R 方、MSE 和 MAE 的最佳结果
print(f"R方最大值 {max_value} 在第 {max_index} 轮")
print(f"最小 RMSE 值 {min_rmse_value} 在第 {min_rmse_index} 轮")
print(f"最小 MAE 值 {min_mae_value} 在第 {min_mae_index} 轮")

# 使用函数简化打印其他指标
print_metrics(max_index, r_squared_series, rmse_series, mae_series)
print_metrics(min_rmse_index, r_squared_series, rmse_series, mae_series)
print_metrics(min_mae_index, r_squared_series, rmse_series, mae_series)




#%% 选择时间步进行测试机绘图
pred_uninverse[-60:, -1] = np.maximum(pred_uninverse[-60:, -1], 0)
df_pred_true = pd.DataFrame({
    'Predict': pred_uninverse[:, -1].flatten(),
    'Real': true_uninverse[:, -1].flatten()
})

df_pred_true.plot(figsize=(12, 4))
plt.title(model_type + ' Result')
plt.show()

df_pred_true = pd.DataFrame({
    'Predict': pred_uninverse[-60:,-1].flatten(),
    'Real': true_uninverse[-60:, -1].flatten()
})



# 绘制图表
df_pred_true.plot(figsize=(12, 4))
# plt.title('Model Result')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
# 将真实值和预测值合并为一个 DataFrame
result_df = pd.DataFrame({'真实值': pred_uninverse[:, -1].flatten(), '预测值': true_uninverse[:, -1].flatten()})
# 保存 DataFrame 到一个 CSV 文件
result_df.to_csv('真实值与预测值7.csv', index=False, encoding='utf-8')
# 打印保存成功的消息
print('真实值和预测值已保存到真实值与预测值.csv文件中。')


