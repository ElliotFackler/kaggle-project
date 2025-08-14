import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
train_df = pd.read_csv('train.csv').drop(columns=['Cabin'])
test_df = pd.read_csv('test.csv').drop(columns=['Cabin']) # We'll use this later for final predictions

# Combine the datasets for easier processing
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Fill missing values in the combined dataset
combined_df['Age'] = combined_df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))  # Fill missing Age with median for each group

# Find the most common port of embarkation and fill missing values.
most_frequent_embarked = combined_df['Embarked'].mode()[0]
print("Most frequent value in Embarked:", most_frequent_embarked)
combined_df['Embarked'] = combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0])

combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median())  # Fill missing Fare with median

combined_df['Sex'] = combined_df['Sex'].replace({'male': 0, 'female': 1})
combined_df['Embarked'] = combined_df['Embarked'].replace({'S': 0, 'Q': 1, 'C': 2})

combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1

combined_df['IsAlone'] = 0
combined_df.loc[combined_df['FamilySize'] == 1, 'IsAlone'] = 1

combined_df['Title'] = combined_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 1, "Mme": 2, "Lady": 2, "Capt": 3, "Sir": 3, "Don": 3, "Dona": 2, "Countess": 2, "Jonkheer": 3}
combined_df['Title'] = combined_df['Title'].map(title_mapping)
combined_df['Title'] = combined_df['Title'].fillna(3)  # Fill any titles that weren't mapped with the 'Rare' group

# Drop the original 'Name', 'Ticket', 'SibSp', and 'Parch' columns
combined_df = combined_df.drop(columns=['Name', 'Ticket', 'SibSp', 'Parch'])

# Now let's re-split the data into train and test sets
train_processed_df = combined_df.iloc[:len(train_df)]
test_processed_df = combined_df.iloc[len(train_df):]
test_processed_df = test_processed_df.drop(columns=['Survived'])

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
sns.countplot(x='Survived', data=train_processed_df)
plt.title('Survival Count (0 = No, 1 = Yes)')
plt.show()

# Explore relationships between features and Survived
# Sex vs. Survived
sns.barplot(x='Sex', y='Survived', data=train_processed_df)
plt.title('Survival Rate by Sex')
plt.show()

# Pclass vs. Survived
sns.barplot(x='Pclass', y='Survived', data=train_processed_df)
plt.title('Survival Rate by Pclass')
plt.show()

# Age distribution (with survival overlay)
plt.figure(figsize=(10, 6))
sns.histplot(train_processed_df[train_processed_df['Survived'] == 0]['Age'].dropna(), bins=30, kde=True, color='red', label='Did Not Survive')
sns.histplot(train_processed_df[train_processed_df['Survived'] == 1]['Age'].dropna(), bins=30, kde=True, color='green', label='Survived')
plt.title('Age Distribution by Survival')
plt.legend()
plt.show()

# Family Size (Family # Aboard) vs. Survived
sns.barplot(x='FamilySize', y='Survived', data=train_processed_df)
plt.title('Survival Rate by Family Size')
plt.show()

# Embarked vs. Survived
sns.barplot(x='Embarked', y='Survived', data=train_processed_df)
plt.title('Survival Rate by Embarked Port')
plt.show()


# Define features (X) and target (y)
X = train_processed_df.drop(columns=['Survived', 'PassengerId'])
y = train_processed_df['Survived'].astype(int)

# Split the training data to validate the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Make predictions on the actual test data
test_predictions = model.predict(test_processed_df.drop(columns=['PassengerId']))

# Create the submission file
submission_df = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': test_predictions})

# Save the submission file to a CSV
submission_df.to_csv('submission.csv', index=False)

print("Submission file created successfully!")
print(submission_df.head())