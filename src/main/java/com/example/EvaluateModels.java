package com.example;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;

public class EvaluateModels {
    public static void main(String[] args) throws Exception {
        // Preprocess data and split into training and test sets
        Instances data = PreprocessData.preprocess("Z:/BSc_MSc/MSc_Courses/FraudDetection_CreditCardTransactions_ML_Java/creditcard.csv");
        Instances[] split = SplitData.splitData(data);
        Instances testData = split[1]; // Use test data for evaluation

        // Load trained models
        Logistic logistic = (Logistic) SerializationHelper.read("logistic.model");
        RandomForest randomForest = (RandomForest) SerializationHelper.read("randomForest.model");
        BayesNet bayesNet = (BayesNet) SerializationHelper.read("bayesNet.model");

        // Evaluate Logistic Regression
        Evaluation evalLogistic = new Evaluation(testData);
        evalLogistic.evaluateModel(logistic, testData);
        System.out.println(evalLogistic.toSummaryString("\nLogistic Regression Results\n======\n", false));

        // Evaluate RandomForest
        Evaluation evalRandomForest = new Evaluation(testData);
        evalRandomForest.evaluateModel(randomForest, testData);
        System.out.println(evalRandomForest.toSummaryString("\nRandomForest Results\n======\n", false));

        // Evaluate BayesNet
        Evaluation evalBayesNet = new Evaluation(testData);
        evalBayesNet.evaluateModel(bayesNet, testData);
        System.out.println(evalBayesNet.toSummaryString("\nBayesNet Results\n======\n", false));
    }
}
