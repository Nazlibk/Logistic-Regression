package nazli.assignment3.util;

import nazli.assignment3.LogisticRegression;
import org.jblas.DoubleMatrix;
import org.math.plot.Plot2DPanel;
import org.math.plot.plotObjects.BaseLabel;

import javax.swing.*;
import java.awt.*;

public class OutputUtil {

    /**
     * This method prints the results.
     *
     * @param results
     */
    public void printResults(LogisticRegression results) {
        System.out.println("Accuracy: " + results.accuracy);
        System.out.println("Recall: " + results.recall);
        System.out.println("Precision: " + results.precision);
        System.out.println("F1score: " + results.f1Score);
        System.out.println("Also, you can see the ROC curve which is opened in another window.");
        System.out.println("If you want to compare the predicted labels with real labels for test set, you can see " +
                "them in Labels.csv file.");

        plotRocCurve(results);
    }

    /**
     * This method plot the ROC curve of results.
     *
     * @param results
     */
    public void plotRocCurve(LogisticRegression results) {
        double[] x = results.rocCurve.getRow(1).toArray();
        double[] y = results.rocCurve.getRow(0).toArray();
        double[] randomX = {0, 0.5, 1};
        double[] randomY = {0, 0.5, 1};
        Plot2DPanel plot = new Plot2DPanel();

        plot.setFixedBounds(0, 0, 1);
        plot.setFixedBounds(1, 0, 1);
        plot.addLinePlot("ROC Curve", Color.BLUE, x, y);
        plot.addLinePlot("Random", Color.RED, randomX, randomY);
        //Add title
        BaseLabel title = new BaseLabel("ROC CURVE", Color.BLACK, 0.5, 1.1);
        title.setFont(new Font("Courier", Font.BOLD, 25));
        plot.addPlotable(title);
        //Add legend
        plot.addLegend("SOUTH");

        //Add axis labels
        plot.setAxisLabels("False Positive Rate (FPR)", "True\nPositive\nRate (TPR)");
        //X Axis properties
        plot.getAxis(0).setLabelPosition(0.5, -0.15);
        plot.getAxis(0).setLabelFont(new Font("Courier", Font.BOLD, 15));
        //Y Axis properties
        plot.getAxis(1).setLabelPosition(-0.05, 0.5);
        plot.getAxis(1).setLabelFont(new Font("Courier", Font.BOLD, 15));

        JFrame frame = new JFrame("ROC Curve");
        frame.setSize(1000, 1000);
        frame.setContentPane(plot);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

    /**
     * This method computes the average of evaluation parameters.
     *
     * @param logisticRegressions
     * @return
     */
    public LogisticRegression averageOfResults(LogisticRegression[] logisticRegressions) {
        LogisticRegression lg = new LogisticRegression();
        //Assign all parameters of lg to 0. This is used in calculating the average.
        lg.f1Score = 0;
        lg.recall = 0;
        lg.precision = 0;
        lg.accuracy = 0;
        lg.confusionMatrix = new DoubleMatrix(new double[logisticRegressions[0].confusionMatrix.rows][logisticRegressions[0].confusionMatrix.columns]);
        for (int j = 0; j < logisticRegressions[0].confusionMatrix.rows; j++) {
            for (int k = 0; k < logisticRegressions[0].confusionMatrix.columns; k++) {
                lg.confusionMatrix.put(j, k, 0);
            }
        }
        lg.rocCurve = new DoubleMatrix(new double[logisticRegressions[0].rocCurve.rows][logisticRegressions[0].rocCurve.columns]);
        for (int j = 0; j < logisticRegressions[0].rocCurve.rows; j++) {
            for (int k = 0; k < logisticRegressions[0].rocCurve.columns; k++) {
                lg.rocCurve.put(j, k, 0);
            }
        }

        //Sum of evaluation parameters of all iterations.
        for (int i = 0; i < logisticRegressions.length; i++) {
            lg.f1Score += logisticRegressions[i].f1Score;
            lg.recall += logisticRegressions[i].recall;
            lg.precision += logisticRegressions[i].precision;
            lg.accuracy += logisticRegressions[i].accuracy;
            for (int j = 0; j < logisticRegressions[0].rocCurve.rows; j++) {
                for (int k = 0; k < logisticRegressions[0].rocCurve.columns; k++) {
                    lg.rocCurve.put(j, k, lg.rocCurve.get(j, k) + logisticRegressions[i].rocCurve.get(j, k));
                }
            }
            for (int j = 0; j < logisticRegressions[0].confusionMatrix.rows; j++) {
                for (int k = 0; k < logisticRegressions[0].confusionMatrix.columns; k++) {
                    lg.confusionMatrix.put(j, k, lg.confusionMatrix.get(j, k) + logisticRegressions[i].confusionMatrix.get(j, k));
                }
            }
        }

        //Average of evaluation parameters of all iterations.
        lg.f1Score /= logisticRegressions.length;
        lg.recall /= logisticRegressions.length;
        lg.precision /= logisticRegressions.length;
        lg.accuracy /= logisticRegressions.length;

        for (int j = 0; j < logisticRegressions[0].rocCurve.rows; j++) {
            for (int k = 0; k < logisticRegressions[0].rocCurve.columns; k++) {
                lg.rocCurve.put(j, k, lg.rocCurve.get(j, k) / logisticRegressions.length);
            }
        }

        for (int j = 0; j < logisticRegressions[0].confusionMatrix.rows; j++) {
            for (int k = 0; k < logisticRegressions[0].confusionMatrix.columns; k++) {
                lg.confusionMatrix.put(j, k, lg.confusionMatrix.get(j, k) / logisticRegressions.length);
            }
        }
        return lg;
    }
}
