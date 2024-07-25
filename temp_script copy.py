import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Load the CSV into a DataFrame
file_path = "kaggle-preprocessed.csv"
df = pd.read_csv(file_path)

# Calculate the correlation matrix
corr_matrix = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix Heatmap")
st.pyplot()

# Line chart
plt.plot(df["PropertyID"], df["SalePrice"])
plt.xlabel("Property ID")
plt.ylabel("Sale Price")
plt.title("Line Chart: Property ID vs Sale Price")
st.pyplot()

# Bar chart
plt.bar(df["Bedrooms"], df["ListPrice"])
plt.xlabel("Bedrooms")
plt.ylabel("List Price")
plt.title("Bar Chart: Bedrooms vs List Price")
st.pyplot()

# Area chart
plt.fill_between(df["YearBuilt"], df["SquareFootage"], alpha=0.5)
plt.xlabel("Year Built")
plt.ylabel("Square Footage")
plt.title("Area Chart: Year Built vs Square Footage")
st.pyplot()