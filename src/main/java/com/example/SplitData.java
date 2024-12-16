package com.example;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class SplitData {
    public static Instances[] splitData(Instances data) throws Exception {
        Resample resample = new Resample();
        resample.setNoReplacement(true);
        resample.setSampleSizePercent(70); // 70% for training
        resample.setRandomSeed(1); // Set a random seed for reproducibility

        // Apply resample to the data
        resample.setInputFormat(data);
        Instances train = Filter.useFilter(data, resample);

        // Create the test set by removing training instances from the original data
        Instances test = new Instances(data);
        for (int i = train.numInstances() - 1; i >= 0; i--) {
            test.delete(i);
        }

        // Print the sizes of the training and test sets for verification
        System.out.println("Training set size: " + train.numInstances());
        System.out.println("Test set size: " + test.numInstances());

        return new Instances[]{train, test};
    }
}
