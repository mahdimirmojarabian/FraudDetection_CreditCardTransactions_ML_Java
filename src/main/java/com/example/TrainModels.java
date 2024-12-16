package com.example;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.Logistic;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class TrainModels {
    public static void main(String[] args) throws Exception {
        // Preprocess data
        Instances data = PreprocessData.preprocess("Z:/BSc_MSc/MSc_Courses/FraudDetection_CreditCardTransactions_ML_Java/creditcard.csv");

        // Split data into training and test sets
        Instances[] split = SplitData.splitData(data);
        Instances trainData = split[0]; // Training data

        // Handle class imbalance in training data
        Instances balancedTrainData = OversampleData.oversample(trainData);

        // Convert numeric attributes to nominal for BayesNet
        NumericToNominal convert = new NumericToNominal();
        convert.setAttributeIndices("first-last");
        convert.setInputFormat(balancedTrainData);
        Instances nominalTrainData = Filter.useFilter(balancedTrainData, convert);

        // Train Logistic Regression
        Logistic logistic = new Logistic();
        logistic.buildClassifier(balancedTrainData);
        SerializationHelper.write("logistic.model", logistic);

        // Train RandomForest
        RandomForest randomForest = new RandomForest();
        randomForest.buildClassifier(balancedTrainData);
        SerializationHelper.write("randomForest.model", randomForest);

        // Train BayesNet with nominal attributes
        BayesNet bayesNet = new BayesNet();
        bayesNet.buildClassifier(nominalTrainData);
        SerializationHelper.write("bayesNet.model", bayesNet);
    }
}
