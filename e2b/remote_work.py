# A simple example of histogram

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization styles
sns.set(style="whitegrid")

df = pd.read_csv('remote_work.csv')

# Display the first few rows of the dataset
print(df.head(10))

# Check for missing values
missing_values = df.isnull().sum()
missing_values[missing_values > 0]

# Plotting the age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=15, kde=True, color='red')
plt.title('Age Distribution of Employees')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('histogram.png')
print("Figure save: histogram.png")