- Project Title: Titanic Survival Prediction
This project is a machine learning solution to the classic Kaggle Titanic competition. It was created to practice and apply fundamental data science techniques, inspired by the concepts discussed in books like Max Tegmark's Life 3.0 and lectures by Lex Fridman.

- Goal
The objective is to predict which passengers survived the Titanic shipwreck based on a variety of features such as age, gender, and ticket class.

- Methodology
I approached this problem using a Random Forest Classifier, a powerful ensemble machine learning algorithm. The process involved the following key steps:

- Data Loading & Cleaning: The raw data was loaded using the pandas library. Missing values in key features like Age and Fare were imputed, and the Cabin column was dropped due to a high number of missing values.

- Feature Engineering: New features were created from the existing data to improve the model's predictive power. This included creating a FamilySize feature by combining SibSp and Parch, as well as extracting titles from the Name column.

- Model Training: A Random Forest Classifier was trained on the processed training data.

- Prediction: The trained model was used to predict survival for the passengers in the test dataset.

- Dependencies
The project relies on the following Python libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn
