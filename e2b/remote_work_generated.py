import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
file_path = 'remote_work.csv'
data = pd.read_csv(file_path)

# Print columns names and its descriptions
# Assuming there's a separate structure or documentation for descriptions
# Here, just printing column names
print('Column Names:'\n)
print(data.columns)

# Assuming descriptions need to be manually defined if they exist
descriptions = {
    'ColumnName1': 'Description of Column 1',
    'ColumnName2': 'Description of Column 2',
    # Add more columns as needed
}

for column, description in descriptions.items():
    print(f'{column}: {description}')

# Print the number of samples
print('\nNumber of samples:', data.shape[0] - 1)

# Remove null samples
nonnull_data = data.dropna()
print('\nRemoved null samples. New number of samples:', nonnull_data.shape[0])

# Plot the distribution by the second column
second_column_name = data.columns[1]
plt.figure(figsize=(10,6))
plt.hist(nonnull_data[second_column_name], bins=20, alpha=0.7)
plt.title(f'Distribution of {second_column_name}')
plt.xlabel(second_column_name)
plt.ylabel('Frequency')
plt.show()