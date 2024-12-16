package com.example;

import static org.junit.Assert.*;

import org.junit.Test;

import weka.core.Instances;

public class TestClass {

    @Test
    public void testPreprocessData() {
        try {
            Instances data = PreprocessData.preprocess("Z:/BSc_MSc/MSc_Courses/FraudDetection_CreditCardTransactions_ML_Java/creditcard.csv");
            assertNotNull(data);
        } catch (Exception e) {
            fail("PreprocessData failed: " + e.getMessage());
        }
    }

    @Test
    public void testSplitData() {
        try {
            Instances data = PreprocessData.preprocess("Z:/BSc_MSc/MSc_Courses/FraudDetection_CreditCardTransactions_ML_Java/creditcard.csv");
            Instances[] split = SplitData.splitData(data);
            assertEquals(2, split.length);
            assertNotNull(split[0]);
            assertNotNull(split[1]);
        } catch (Exception e) {
            fail("SplitData failed: " + e.getMessage());
        }
    }

    @Test
    public void testOversampleData() {
        try {
            Instances data = PreprocessData.preprocess("Z:/BSc_MSc/MSc_Courses/FraudDetection_CreditCardTransactions_ML_Java/creditcard.csv");
            Instances oversampledData = OversampleData.oversample(data);
            assertNotNull(oversampledData);
        } catch (Exception e) {
            fail("OversampleData failed: " + e.getMessage());
        }
    }
}
