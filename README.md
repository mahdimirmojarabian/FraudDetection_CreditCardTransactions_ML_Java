# Fraud Detection in Credit Card Transactions using ML Techniques (in Java)

This project leverages machine learning models (Logistic Regression, RandomForest, and BayesNet models) to detect fraudulent credit card transactions. The models are trained and evaluated using a dataset of credit card transactions from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/data).

## Project Structure

```plaintext

├── .vscode/ (not included in repository)
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── example/
│   │               ├── EvaluateModels.java (Evaluates the trained models using the test data)
│   │               ├── OversampleData.java (Handles class imbalance using SMOTE)
│   │               ├── PreprocessData.java (Loads and preprocesses the dataset)
│   │               ├── SplitData.java      (Splits the dataset into training and test sets)
│   │               └── TrainModels.java    (Trains and saves the machine learning models)
│   └── test/
│       └── java/
│           └── com/
│               └── TestClass.java          (Contains unit tests for the project.)
|
├── bayesNet.model
├── logistic.model
├── randomForest.model
├── .gitignore
├── LICENSE
├── pom.xml
├── README.md
└── creditcard.csv (not included in repository)

## Project Structure
- Java (23.0.1)
- Apache Maven (3.9.9)
- Weka

## Setup
1- Clone the Repository:
`git clone https://github.com/mahdimirmojarabian/FraudDetection_CreditCardTransactions_ML_Java.git`

2. Build the Project:
Build the project using Maven to resolve dependencies and compile the code:

`mvn clean package`

3. Dataset:
Download the dataset from from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/data).

4. Training the Models:
Run the training script to preprocess the data, split it into training and test sets, handle class imbalance, train the models, and save them:

`java -jar target/fraud-detection-1.0-SNAPSHOT.jar`

5. Evaluating the Models:
Run the evaluation script to load the trained models and evaluate them on the test set:
`java -cp target/fraud-detection-1.0-SNAPSHOT.jar com.example.EvaluateModels`

## Results:

## Training and Test Set Sizes

- **Training set size:** 199,364
- **Test set size:** 85,443

## Logistic Regression Results
======

- **Correctly Classified Instances:** 83,499 (97.7248%)
- **Incorrectly Classified Instances:** 1,944 (2.2752%)
- **Kappa statistic:** 0.0877
- **Mean absolute error:** 0.0634
- **Root mean squared error:** 0.1395
- **Relative absolute error:** 2499.3835%
- **Root relative squared error:** 392.5173%
- **Total Number of Instances:** 85,443

## RandomForest Results
======

- **Correctly Classified Instances:** 85,430 (99.9848%)
- **Incorrectly Classified Instances:** 13 (0.0152%)
- **Kappa statistic:** 0.9389
- **Mean absolute error:** 0.001
- **Root mean squared error:** 0.0127
- **Relative absolute error:** 41.0895%
- **Root relative squared error:** 35.8112%
- **Total Number of Instances:** 85,443

## BayesNet Results
======

- **Correctly Classified Instances:** 108 (0.1264%)
- **Incorrectly Classified Instances:** 85,335 (99.8736%)
- **Kappa statistic:** 0
- **Mean absolute error:** 0.9987
- **Root mean squared error:** 0.9994
- **Relative absolute error:** 39375.3464%
- **Root relative squared error:** 2812.719%
- **Total Number of Instances:** 85,443
