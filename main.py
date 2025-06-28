import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv') # We'll use this later for final predictions

# Combine the datasets for easier processing
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Fill missing values in the combined dataset
combined_df['Age'] = combined_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))  # Fill missing Age with median for each group

# Display the first few rows of the training data
print("Train Data Head:")
print(train_df.head())

print("\nTest Data Head:")
print(test_df.head()) # Note: 'Survived' column is missing in test_df

# Get basic information about the data
print("\nTrain Data Info:")
train_df.info()

print("\nMissing values in Train Data:")
print(train_df.isnull().sum())

print("\nMissing values in Test Data:")
print(test_df.isnull().sum())

# Describe numerical columns
print("\nTrain Data Description (Numerical):")
print(train_df.describe())

# Describe categorical columns (can use include='object' for string columns)
print("\nTrain Data Description (Categorical):")
print(train_df.describe(include=['object']))

# Let's visualize the target variable (Survived)
sns.countplot(x='Survived', data=train_df)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()

# Explore relationships between features and Survived
# Sex vs. Survived
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()

# Pclass vs. Survived
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Pclass')
plt.show()

# Age distribution (with survival overlay)
plt.figure(figsize=(10, 6))
sns.histplot(train_df[train_df['Survived'] == 0]['Age'].dropna(), bins=30, kde=True, color='red', label='Did Not Survive')
sns.histplot(train_df[train_df['Survived'] == 1]['Age'].dropna(), bins=30, kde=True, color='green', label='Survived')
plt.title('Age Distribution by Survival')
plt.legend()
plt.show()

# SibSp (Siblings/Spouses Aboard) vs. Survived
sns.barplot(x='SibSp', y='Survived', data=train_df)
plt.title('Survival Rate by Siblings/Spouses Aboard')
plt.show()

# Parch (Parents/Children Aboard) vs. Survived
sns.barplot(x='Parch', y='Survived', data=train_df)
plt.title('Survival Rate by Parents/Children Aboard')
plt.show()

# Embarked vs. Survived
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title('Survival Rate by Embarked Port')
plt.show()