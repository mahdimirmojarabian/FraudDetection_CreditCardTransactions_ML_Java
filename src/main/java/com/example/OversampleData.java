package com.example;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class OversampleData {
    public static Instances oversample(Instances data) throws Exception {
        Resample resample = new Resample();
        resample.setBiasToUniformClass(1.0);  // Set bias to 1 to oversample minority class
        resample.setInputFormat(data);
        Instances oversampledData = Filter.useFilter(data, resample);

        return oversampledData;
    }
}
