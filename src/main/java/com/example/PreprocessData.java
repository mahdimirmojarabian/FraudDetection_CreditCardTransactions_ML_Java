package com.example;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class PreprocessData {
    public static Instances preprocess(String filepath) throws Exception {
        DataSource source = new DataSource(filepath);
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Convert class attribute from numeric to nominal
        NumericToNominal convert = new NumericToNominal();
        convert.setAttributeIndices("" + (data.classIndex() + 1));
        convert.setInputFormat(data);
        Instances nominalData = Filter.useFilter(data, convert);

        // Normalize data
        Normalize normalize = new Normalize();
        normalize.setInputFormat(nominalData);
        Instances normalizedData = Filter.useFilter(nominalData, normalize);

        // Convert nominal attributes to binary
        NominalToBinary nominalToBinary = new NominalToBinary();
        nominalToBinary.setInputFormat(normalizedData);
        Instances binaryData = Filter.useFilter(normalizedData, nominalToBinary);

        return binaryData;
    }
}
