Here is a Python script that fulfills the requirements:

```python
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
filename = 'kaggle-preprocessed.csv'
df = pd.read_csv(filename)

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Plot line chart
st.line_chart(df[['PropertyID', 'SalePrice']])

# Plot bar chart
st.bar_chart(df[['YearBuilt', 'ListPrice']])

# Plot area chart
st.area_chart(df[['SquareFootage', 'Bathrooms']])

# Remove warning for deprecation
st.set_option('deprecation.showPyplotGlobalUse', False)
```

This script uses the `pandas` library to load the CSV file into a DataFrame. It then calculates the correlation matrix between the columns using the `corr()` method. The correlation matrix is plotted as a heatmap using the `seaborn` library and `matplotlib`. Additionally, it uses `streamlit` to plot line, bar, and area charts using two columns from the DataFrame. The warning for deprecation is disabled using `st.set_option()`.