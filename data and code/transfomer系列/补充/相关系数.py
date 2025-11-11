import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import os
new_working_directory = r"D:\毕业论文\傅立叶卷积_itransformer\傅立叶卷积_itransformer\transfomer系列"
os.chdir(new_working_directory)

# Load the dataset
data = pd.read_excel('./数据集/内蒙古.xlsx')
# data = data.iloc[:35586,:]
# data = pd.read_excel('./数据集/风电2019.xlsx')  
# Extract features and target
features = data.drop(columns=['date', 'Target'])  # Exclude 'date' and 'Target' columns
target = data['Target']

# Calculate MIC for each feature against the target
mic_values = []
for column in features.columns:
    mic = mutual_info_regression(features[[column]], target, random_state=0)
    mic_values.append(mic[0])

# Create a DataFrame for visualization
mic_df = pd.DataFrame(mic_values, index=features.columns, columns=['MIC_with_Target'])

# Generate a full MIC heatmap for feature-to-feature and feature-to-target MICs
# Combine features and target into one dataset for pairwise MIC computation
full_data = features.copy()
full_data['Target'] = target

# Calculate the pairwise MICs
pairwise_mic = pd.DataFrame(index=full_data.columns, columns=full_data.columns)

for col1 in full_data.columns:
    for col2 in full_data.columns:
        mic = mutual_info_regression(full_data[[col1]], full_data[col2], random_state=0)
        pairwise_mic.loc[col1, col2] = mic[0]

pairwise_mic = pairwise_mic.astype(float)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    pairwise_mic,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={"label": "MIC Value"},
)
plt.title("MIC Relationship Analysis")
plt.show()
