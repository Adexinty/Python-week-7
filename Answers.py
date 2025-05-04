# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset from an online source
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
df = pd.read_csv(url)

# Task 1: Explore the Dataset
print("First 5 rows:")  # Display the first few rows of the dataset
print(df.head())

print("\nMissing values:")  # Check for missing values in the dataset
print(df.isnull().sum())

# Drop missing values if any
df.dropna(inplace=True)

# Task 2: Basic Data Analysis
print("\nSummary statistics:")  # Print summary statistics of the dataset
print(df.describe())

print("\nMean petal length by species:")  # Calculate the average petal length for each species
print(df.groupby("species")["petal_length"].mean())

# Task 3: Data Visualizations

# Add a simulated "day" column to create a time-based visualization
df["day"] = range(1, len(df) + 1)

# Line chart: Petal length over time by species
plt.figure()
sns.lineplot(data=df, x="day", y="petal_length", hue="species")
plt.title("Petal Length Over Time")
plt.xlabel("Day")
plt.ylabel("Petal Length")
plt.show()

# Bar chart: Average petal length by species
plt.figure()
sns.barplot(data=df, x="species", y="petal_length", ci=None)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length")
plt.show()

# Histogram: Distribution of sepal width
plt.figure()
sns.histplot(df["sepal_width"], bins=15, kde=True)
plt.title("Sepal Width Distribution")
plt.xlabel("Sepal Width")
plt.show()

# Scatter plot: Relationship between sepal length and petal length
plt.figure()
sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
