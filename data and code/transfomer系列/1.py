import pandas as pd

df = pd.DataFrame({
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'age': [20, 21, 22, 23, 24]
})


# 计算每组的平均值并广播到每行
# df['Group_Mean'] = df.groupby('Group')['Value'].transform('mean')
df.groupby('gender').agg({'age':'mean'})
