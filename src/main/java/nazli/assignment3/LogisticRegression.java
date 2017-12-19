package nazli.assignment3;

import nazli.assignment3.util.FileUtility;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class LogisticRegression {

    //Algorithm settings
    final double alpha = 0.01;    //Learning rate
    final double tolerance = 1e-5;      //Tolerance to determine convergence
    final int maxiteration = 1000;    //Maximum number of iterations (in case convergence is not reached)
    final double datasetSpilittingRate = 2.0 / 3.0;    //Splitting rate of data set to training set adn test set
    final int lambda = 4;        //Parameter used in regularization (for avoiding overfitting)
    public double precision;
    public double recall;
    public double f1Score;
    public double accuracy;
    public DoubleMatrix rocCurve;      //This matrix has the results for ROC curve. It has two rows.
    // First row has the values for TP and the second row has the values for FP.
    public double fpRate;
    public double fnRate;
    String[] types = {"LongEaredOwl", "SnowyOwl", "BarnOwl"};
    //evaluation parameters
    public DoubleMatrix confusionMatrix = new DoubleMatrix(new double[types.length][types.length]);   //This is the main confusion matrix for logistic regression
    List<Owl> owls = new ArrayList<>();
    DoubleMatrix theta = new DoubleMatrix(new double[types.length][5]);
    DoubleMatrix h;
    DoubleMatrix y;
    DoubleMatrix j = new DoubleMatrix(new double[types.length][maxiteration]);

    public LogisticRegression(List<Owl> owls) {
        //First, constructor normalized features using 0-1 normalizing.
        List<Owl> normalizedOwls = new ArrayList<>();
        DoubleMatrix features = getFeatures(owls);
        DoubleMatrix normalizedFeatures = normalized(features);
        for (int i = 0; i < owls.size(); i++) {
            Owl owl = new Owl();
            owl.setBodyLength(normalizedFeatures.get(i, 1));
            owl.setWingLength(normalizedFeatures.get(i, 2));
            owl.setBodyWidth(normalizedFeatures.get(i, 3));
            owl.setWingWidth(normalizedFeatures.get(i, 4));
            owl.setType(owls.get(i).getType());
            normalizedOwls.add(owl);
        }
        this.owls = normalizedOwls;
    }

    public LogisticRegression() {
    }

    /**
     * This is the main method which runs logistic regression. Other methods of this class invoke from this method.
     *
     * @return
     * @throws AssignmentException
     */
    public LogisticRegression run() throws AssignmentException {
        FileUtility fileUtility = new FileUtility();
        //Split data to training and test set
        List<Owl> trainingSetOwls = new ArrayList<>();
        List<Owl> testSetOwls = new ArrayList<>();
        for (Owl owl : owls) {
            double random = Math.random();
            if (random < datasetSpilittingRate) {
                trainingSetOwls.add(owl);
            } else {
                testSetOwls.add(owl);
            }
        }
        DoubleMatrix testSet = getFeatures(testSetOwls);
        DoubleMatrix trainingSet = getFeatures(trainingSetOwls);
        this.h = new DoubleMatrix(new double[types.length][testSet.rows]);
        this.y = new DoubleMatrix(new double[types.length][testSet.rows]);
        DoubleMatrix subConfusionMatrix = new DoubleMatrix(new double[types.length][4]);       //This is a confusion matrix[3][4] for
        // all three types of owls. This sunConfusionMatrix is used for training logistic regression for each of types
        // and find the best theta fit our input. Then we have overall typesConfusionMatrixes for logistic regression which
        // represents the confusion matrix regardless of their types.

        for (int i = 0; i < types.length; i++) {
            DoubleMatrix theta = DoubleMatrix.zeros(trainingSet.columns);
            DoubleMatrix yTraining = binaryY(types[i], trainingSetOwls);
            DoubleMatrix yTest = binaryY(types[i], testSetOwls).transpose();
            DoubleMatrix j = DoubleMatrix.zeros(maxiteration);

            //Train
            double deltaTheta;
            int iteration = 0;
            do {
                DoubleMatrix thetaGradient = regularizedUpdateTheta(theta, trainingSet, yTraining);
                deltaTheta = delta(theta, thetaGradient);
                theta = theta.sub(thetaGradient);
                this.j.put(i, iteration, regularizedCostFunction(theta, trainingSet, yTraining));
                iteration++;
            } while (iteration < maxiteration || deltaTheta < tolerance);
            this.theta.putRow(i, theta);

            //Test
            this.h.putRow(i, h(this.theta.getRow(i).transpose(), testSet));
            this.y.putRow(i, hToY(this.h.getRow(i)));
            subConfusionMatrix.putRow(i, typesConfusionMatrixes(this.y.getRow(i), yTest));
        }

        //Predict labels
        DoubleMatrix finalHypothesisValues = finalH(this.theta, testSet);
        String[] predictedLabels = finalOutputLabels(finalHypothesisValues.transpose());
        fileUtility.writeOnCsvFile(predictedLabels, testSetOwls);       //Writes the predicted and real labels in the Labels.csv file.
        evaluation(predictedLabels, testSetOwls, finalHypothesisValues);    //Evaluates the algorithm and calculates evaluation parameters.

        return this;
    }

    /**
     * This function calculates the TP and FP values for plotting ROC curve.
     *
     * @param h
     * @param types
     * @param testSetOwls
     * @return a matrix with two rows. First row has the results for TPs and the second row has the results for FPs.
     */
    DoubleMatrix rocCurveValues(DoubleMatrix h, String[] types, List<Owl> testSetOwls) {
        DoubleMatrix rocCurveResults;
        double[][] normalizedHValues = new double[h.rows][h.columns];
        for (int i = 0; i < types.length; i++) {
            for (int j = 0; j < h.columns; j++) {
                normalizedHValues[i][j] = h.get(i, j) / h.getColumn(j).sum();
            }
        }

        //This part of code classifies one class vs all other classes and find TP and FP for each type.
        // Then, get the average of all types because at the end we should just have on value for TP
        // and one value for FP in each threshold.

        List<Double> thresholds = new ArrayList<>();
        double threshold = 0;
        //Thresholds are started from 0 to 1 with steps 0.01
        double thresholdStep = 0.01;
        while (threshold <= 1) {
            thresholds.add(threshold);
            threshold += thresholdStep;
        }
        rocCurveResults = new DoubleMatrix(new double[2][thresholds.size()]);
        for (int k = 0; k < thresholds.size(); k++) {
            double[] curveValues = new double[2];
            for (int i = 0; i < types.length; i++) {
                DoubleMatrix yTest = binaryY(types[i], testSetOwls).transpose();
                DoubleMatrix predictedY = new DoubleMatrix(new double[h.columns]);
                for (int j = 0; j < h.columns; j++) {
                    if (normalizedHValues[i][j] <= thresholds.get(k)) {
                        predictedY.put(j, 0);
                    } else {
                        predictedY.put(j, 1);
                    }
                }
                DoubleMatrix confusionMat = typesConfusionMatrixes(predictedY.transpose(), yTest);
                Double tpRate = confusionMat.get(3) / (confusionMat.get(3) + confusionMat.get(2));
                Double fpRate = confusionMat.get(1) / (confusionMat.get(1) + confusionMat.get(0));

                curveValues[0] += tpRate;
                curveValues[1] += fpRate;
            }
            rocCurveResults.put(0, k, curveValues[0] / types.length);
            rocCurveResults.put(1, k, curveValues[1] / types.length);
        }
        return rocCurveResults;
    }

    /**
     * This method provides all the parameters needed for evaluation of the algorithm.
     *
     * @param predictedLabels
     * @param testSet
     */
    void evaluation(String[] predictedLabels, List<Owl> testSet, DoubleMatrix h) {
        this.confusionMatrix = confusionMatrix(predictedLabels, testSet);
        this.precision = getPrecision(this.confusionMatrix);
        this.recall = getRecall(this.confusionMatrix);
        this.f1Score = getF1Score(confusionMatrix);
        this.accuracy = getAccuracy(confusionMatrix);
        this.rocCurve = rocCurveValues(h, types, testSet);
        this.fpRate = getFPRate(confusionMatrix);
        this.fnRate = getFNRate(confusionMatrix);
    }

    /**
     * This method calculates the precision using TP/TP + FP for each type of owls.
     * Overall result is the average of precision of all types.
     *
     * @param confusionMatrix
     * @return precision
     */
    double getPrecision(DoubleMatrix confusionMatrix) {
        double[] precisions = new double[confusionMatrix.rows];
        for (int i = 0; i < confusionMatrix.rows; i++) {
            if (confusionMatrix.getColumn(i).sum() == 0) {
                precisions[i] = 0;
            } else {
                precisions[i] = confusionMatrix.get(i, i) / confusionMatrix.getColumn(i).sum();
            }
        }
        return new DoubleMatrix(precisions).mean();
    }

    /**
     * This method calculates the accuracy using (TP + TN)/(TP + TN + FP + FN) for each type of owls.
     * Overall result is the average of accuracy of all types.
     *
     * @param confusionMatrix
     * @return accuracy
     */
    double getAccuracy(DoubleMatrix confusionMatrix) {
        double[] accuracies = new double[confusionMatrix.rows];
        for (int i = 0; i < confusionMatrix.rows; i++) {
            double TN = confusionMatrix.sum() - confusionMatrix.getRow(i).sum() - confusionMatrix.getColumn(i)
                                                                                                 .sum() + confusionMatrix
                    .get(i, i);
            accuracies[i] = (confusionMatrix.get(i, i) + TN) / confusionMatrix.sum();
        }
        return new DoubleMatrix(accuracies).mean();
    }

    /**
     * This method calculates the FN rate for each type using FN/(FN + TP). then, return the average of FN rates of types as result.
     *
     * @param confusionMatrix
     * @return FN rate
     */
    double getFNRate(DoubleMatrix confusionMatrix) {
        double[] fnRates = new double[confusionMatrix.rows];
        for (int i = 0; i < confusionMatrix.rows; i++) {
            fnRates[i] = (confusionMatrix.getRow(i).sum() - confusionMatrix.get(i, i)) / (confusionMatrix.getRow(i)
                                                                                                         .sum());
        }
        return new DoubleMatrix(fnRates).mean();
    }

    /**
     * This method calculates the FP rate for each type using FP/(FP + TP). Then, return the average of FP rates of types as result.
     *
     * @param confusionMatrix
     * @return FP rate
     */
    double getFPRate(DoubleMatrix confusionMatrix) {
        double[] fpRates = new double[confusionMatrix.rows];
        for (int i = 0; i < confusionMatrix.rows; i++) {
            fpRates[i] = (confusionMatrix.getColumn(i)
                                         .sum() - confusionMatrix.get(i, i)) / (confusionMatrix.getColumn(i).sum());
        }
        return new DoubleMatrix(fpRates).mean();
    }

    /**
     * This method calculates the F1score using 2*getPrecision*getRecall/(getPrecision + getRecall).
     *
     * @param confusionMatrix
     * @return F1score
     */
    double getF1Score(DoubleMatrix confusionMatrix) {
        double precision = getPrecision(confusionMatrix);
        double recall = getRecall(confusionMatrix);
        return (2 * precision * recall) / (precision + recall);
    }

    /**
     * This method calculates the recall using TP/(TP + FN) for each type of owls.
     * Overall result is the average of recall for all types.
     *
     * @param confusionMatrix
     * @return recall
     */
    double getRecall(DoubleMatrix confusionMatrix) {
        double[] recalls = new double[confusionMatrix.rows];
        for (int i = 0; i < confusionMatrix.rows; i++) {
            if (confusionMatrix.getRow(i).sum() == 0) {
                recalls[i] = 0;
            } else {
                recalls[i] = confusionMatrix.get(i, i) / (confusionMatrix.getRow(i).sum());
            }
        }
        return new DoubleMatrix(recalls).mean();
    }

    /**
     * This function gets x matrix and using 0-1 normalization, returned the normalized matrix.
     *
     * @param x
     * @return using 0-1 normalization, it returns the normalized values of x
     */
    DoubleMatrix normalized(DoubleMatrix x) {
        double[] min = new double[x.columns];
        double[] max = new double[x.columns];
        DoubleMatrix result = new DoubleMatrix(new double[x.rows][x.columns]);
        for (int i = 0; i < x.columns; i++) {
            double[] values = x.getColumn(i).toArray();
            DescriptiveStatistics descriptiveStatistics = new DescriptiveStatistics(values);
            min[i] = descriptiveStatistics.getMin();
            max[i] = descriptiveStatistics.getMax();
        }
        for (int i = 0; i < x.rows; i++) {
            for (int j = 0; j < x.columns; j++) {
                result.put(i, j, (x.get(i, j) - min[j]) / (max[j] - min[j]));
            }
        }
        return result;
    }

    /**
     * This function gets test set and final thetas for all 3 types. Calculates the h = 1/(1 + exp(-x*theta)).
     * This h shows the probability for each type.
     *
     * @param theta
     * @param x
     * @return
     */
    DoubleMatrix finalH(DoubleMatrix theta, DoubleMatrix x) {
        DoubleMatrix finalHypothesis = DoubleMatrix.zeros(theta.rows, x.rows);
        for (int i = 0; i < theta.rows; i++) {
            finalHypothesis.putRow(i, h(theta.getRow(i).transpose(), x));
        }
        return finalHypothesis;
    }

    /**
     * This function gets h(theta) for all types and give back the final type.
     * Final type is the one which has greater h(theta) (probability) among the others.
     *
     * @param h
     * @return Array of string which represents our predictions for each data in the test set.
     */
    String[] finalOutputLabels(DoubleMatrix h) {
        String[] finalResults = new String[h.rows];
        for (int i = 0; i < h.rows; i++) {
            double max = 0;
            int maxIndex = 0;
            for (int j = 0; j < h.columns; j++) {
                if (h.get(i, j) > max) {
                    max = h.get(i, j);
                    maxIndex = j;
                }
            }
            if (maxIndex == 0) {
                finalResults[i] = "LongEaredOwl";
            } else if (maxIndex == 1) {
                finalResults[i] = "SnowyOwl";
            } else if (maxIndex == 2) {
                finalResults[i] = "BarnOwl";
            }
        }
        return finalResults;
    }

    /**
     * This function provides confusion matrix.
     *
     * @param predictedY is the results of our model.
     * @param actualY    is real results.
     * @return This is a 3*4 matrix. 3 represents confusion matrix for each of our 3 types of owls and 4 represents TN, FP, FN, TP.
     * So, in each row index 0 = TN, 1 = FP, 2 = FN and 3 = TP. |TN|FP|FN|TP|
     */
    DoubleMatrix typesConfusionMatrixes(DoubleMatrix predictedY, DoubleMatrix actualY) {
        DoubleMatrix result = DoubleMatrix.zeros(4);
        for (int i = 0; i < predictedY.columns; i++) {
            if (predictedY.get(i) == 0 && actualY.get(i) == 0) {
                result.put(0, result.get(0) + 1);
            } else if (predictedY.get(i) == 1 && actualY.get(i) == 0) {
                result.put(1, result.get(1) + 1);
            } else if (predictedY.get(i) == 0 && actualY.get(i) == 1) {
                result.put(2, result.get(2) + 1);
            } else if (predictedY.get(i) == 1 && actualY.get(i) == 1) {
                result.put(3, result.get(3) + 1);
            }
        }
        return result;
    }

    /**
     * This is the confusion matrix of logistic regression.
     *
     * @param predictedLabels These are the labels that our model predict for test set.
     * @param testSet         This is the test set where we should extract the labels for each data and compare them with predicted labels.
     * @return 3*3 matrix which represents confusion matrix. columns represent the predicted values and rows represents actual values.
     */
    DoubleMatrix confusionMatrix(String[] predictedLabels, List<Owl> testSet) {
        //types order in temporary confusion matrix: LongEaredOwl, SnowyOwl, BarnOwl
        DoubleMatrix confusionMatrix = DoubleMatrix.zeros(types.length, types.length);
        for (int j = 0; j < predictedLabels.length; j++) {
            if (testSet.get(j).getType().equals("LongEaredOwl") && predictedLabels[j].equals("LongEaredOwl")) {
                confusionMatrix.put(0, 0, confusionMatrix.get(0, 0) + 1);
            } else if (testSet.get(j).getType().equals("LongEaredOwl") && predictedLabels[j].equals("SnowyOwl")) {
                confusionMatrix.put(0, 1, confusionMatrix.get(0, 1) + 1);
            } else if (testSet.get(j).getType().equals("LongEaredOwl") && predictedLabels[j].equals("BarnOwl")) {
                confusionMatrix.put(0, 2, confusionMatrix.get(0, 2) + 1);
            } else if (testSet.get(j).getType().equals("SnowyOwl") && predictedLabels[j].equals("LongEaredOwl")) {
                confusionMatrix.put(1, 0, confusionMatrix.get(1, 0) + 1);
            } else if (testSet.get(j).getType().equals("SnowyOwl") && predictedLabels[j].equals("SnowyOwl")) {
                confusionMatrix.put(1, 1, confusionMatrix.get(1, 1) + 1);
            } else if (testSet.get(j).getType().equals("SnowyOwl") && predictedLabels[j].equals("BarnOwl")) {
                confusionMatrix.put(1, 2, confusionMatrix.get(1, 2) + 1);
            } else if (testSet.get(j).getType().equals("BarnOwl") && predictedLabels[j].equals("LongEaredOwl")) {
                confusionMatrix.put(2, 0, confusionMatrix.get(2, 0) + 1);
            } else if (testSet.get(j).getType().equals("BarnOwl") && predictedLabels[j].equals("SnowyOwl")) {
                confusionMatrix.put(2, 1, confusionMatrix.get(2, 1) + 1);
            } else if (testSet.get(j).getType().equals("BarnOwl") && predictedLabels[j].equals("BarnOwl")) {
                confusionMatrix.put(2, 2, confusionMatrix.get(2, 2) + 1);
            }
        }
        return confusionMatrix;
    }

    /**
     * This function get h (hypothesis) which shows the probability of getting 1 (or preferred type)
     * and return y for this hypothesis. All elements of y is 1 or 0.
     *
     * @param h
     * @return y for the hypothesis which is 1 or 0.
     */
    DoubleMatrix hToY(DoubleMatrix h) {
        DoubleMatrix result = DoubleMatrix.zeros(h.columns);
        for (int i = 0; i < h.columns; i++) {
            if (h.get(i) < 0.5) {
                result.put(i, 0);
            } else {
                result.put(i, 1);
            }
        }
        return result;
    }

    /**
     * This function gets two theta and calculate ||theta1| - |theta2||.
     *
     * @param theta1
     * @param theta2
     * @return ||theta1| - |theta2||
     */
    double delta(DoubleMatrix theta1, DoubleMatrix theta2) {
        double sum1 = 0;
        double sum2 = 0;
        for (int i = 0; i < theta1.rows; i++) {
            sum1 += Math.pow(theta1.get(i), 2);
        }
        for (int i = 0; i < theta2.rows; i++) {
            sum2 += Math.pow(theta2.get(i), 2);
        }
        return Math.abs(sum1 - sum2);
    }

    /**
     * This is the cost function.
     *
     * @param theta
     * @param x
     * @param y
     * @return This function calculates the value of: (1/number of rows in dataset)(-y.transpose()*log(h)-(1-y).transpose()*log(1-h))
     */
    double costFunction(DoubleMatrix theta, DoubleMatrix x, DoubleMatrix y) {
        double result;
        DoubleMatrix h = h(theta, x);
        DoubleMatrix tmpLogH = new DoubleMatrix(new double[x.rows]);
        DoubleMatrix tmpLog1_H = new DoubleMatrix(new double[x.rows]);
        DoubleMatrix ones = DoubleMatrix.ones(y.columns);
        for (int i = 0; i < x.rows; i++) {
            tmpLogH.put(i, Math.log(h.get(i)));
            tmpLog1_H.put(i, Math.log(1 - h.get(i)));
        }
        result = (1.0 / x.rows) * (y.neg()
                                    .transpose()
                                    .mmul(tmpLogH)
                                    .sub((ones.sub(y)).transpose().mmul(tmpLog1_H))).toArray()[0];
        return result;
    }

    /**
     * This is the regularized cost function. Regularization help avoid overfitting.
     *
     * @param theta
     * @param x
     * @param y
     * @return This function calculates the value of:
     * (1/number of rows in dataset)(-y.transpose()*log(h)-(1-y).transpose()*log(1-h))+(lambda/(2 * number of rows in dataset))*sigma(theta.pow(2))
     */
    double regularizedCostFunction(DoubleMatrix theta, DoubleMatrix x, DoubleMatrix y) {
        double result = costFunction(theta, x, y);
        double sum = 0;
        for (int i = 1; i < theta.rows; i++) {
            sum += Math.pow(theta.get(i), 2);
        }
        return result + (lambda * sum) / (2 * x.rows);
    }

    /**
     * This is the function which update theta in each step.
     *
     * @param theta
     * @param x
     * @param y
     * @return This function uses the equation below and calculate the new values for theta and return the new theta values:
     * theta = theta - (alpha/number of rows in dataset)*x.transpose()*(h-y)
     */
    DoubleMatrix updateTheta(DoubleMatrix theta, DoubleMatrix x, DoubleMatrix y) {
        return x.transpose().mmul(h(theta, x).sub(y)).mul(alpha / x.rows);
    }

    /**
     * This is the function which update theta using regularization in each step.
     *
     * @param theta
     * @param x
     * @param y
     * @return This function uses the equation below and calculate the new values for theta and return the new theta values:
     * theta = theta - (alpha/number of rows in dataset)*x.transpose()*(h-y) - (alpha*lambda*theta)/number of rows in dataset
     */
    DoubleMatrix regularizedUpdateTheta(DoubleMatrix theta, DoubleMatrix x, DoubleMatrix y) {
        DoubleMatrix result = updateTheta(theta, x, y);
        for (int i = 1; i < theta.rows; i++) {
            double tmp = (alpha * lambda * theta.get(i)) / x.rows;
            result.put(i, result.get(i) - tmp);
        }
        return result;
    }

    /**
     * This function gets list of owls as function as extract the features.
     *
     * @param owls
     * @return extracted features from list of owls.
     */
    DoubleMatrix getFeatures(List<Owl> owls) {
        DoubleMatrix input = new DoubleMatrix(new double[owls.size()][5]);
        for (int i = 0; i < owls.size(); i++) {
            input.put(i, 0, 1);
            input.put(i, 1, owls.get(i).getBodyLength());
            input.put(i, 2, owls.get(i).getWingLength());
            input.put(i, 3, owls.get(i).getBodyWidth());
            input.put(i, 4, owls.get(i).getWingWidth());
        }
        return input;
    }

    /**
     * This function (h) shows the hypothesis. z = theta0+theta1*x1+...+thetan*xn
     * h = 1/(1 + exp(-x*theta))
     *
     * @param theta
     * @param x
     * @return h = 1/(1 + exp(-x*theta))
     */
    DoubleMatrix h(DoubleMatrix theta, DoubleMatrix x) {
        DoubleMatrix result = new DoubleMatrix(new double[x.rows]);
        DoubleMatrix tmp = x.mmul(theta).transpose();
        for (int i = 0; i < x.rows; i++) {
            result.put(i, 1.0 / (1 + Math.exp(-tmp.get(0, i))));
        }
        return result;
    }

    /**
     * Logistic regression output is 1 or 0 and it can classify all data to two classes. But we have 3 types of owls.
     * So, we have to assign these classes to 1 and 0. So, we have to run the same program 3 times and
     * find the best theta for each class. For getting y (labels) for class LongEaredOwl, 1 is assigned to LongEaredOwl
     * and 0 to others as new labels.
     *
     * @param type
     * @param owls
     * @return
     */
    DoubleMatrix binaryY(String type, List<Owl> owls) {
        DoubleMatrix y = new DoubleMatrix(new double[owls.size()]);
        for (int i = 0; i < owls.size(); i++) {
            if (owls.get(i).getType().equals(type)) {
                y.put(i, 1);
            } else {
                y.put(i, 0);
            }
        }
        return y;
    }
}
