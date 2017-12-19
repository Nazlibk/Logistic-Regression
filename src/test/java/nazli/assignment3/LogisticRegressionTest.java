package nazli.assignment3;

import org.jblas.DoubleMatrix;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

public class LogisticRegressionTest {
    LogisticRegression lg;
    List<Owl> owls;

    @Before
    public void setup() {
        Owl owl1 = new Owl(0.87, 0.65, 0.43, 0.9, "LongEaredOwl");
        Owl owl2 = new Owl(0.71, 0.78, 0.65, 0.85, "LongEaredOwl");
        Owl owl3 = new Owl(0.54, 0.67, 0.45, 0.13, "SnowyOwl");
        Owl owl4 = new Owl(0.12, 0.7, 0.71, 0.56, "BarnOwl");
        Owl owl5 = new Owl(0.192, 0.6, 0.671, 0.856, "BarnOwl");
        owls = new ArrayList<>();
        owls.add(owl1);
        owls.add(owl2);
        owls.add(owl3);
        owls.add(owl4);
        owls.add(owl5);
        lg = new LogisticRegression(owls);
    }

    @Test
    public void test_h_while_theta_equals_to_zero() {
        DoubleMatrix x = new DoubleMatrix(new double[][]{{1, 1, 2}, {1, 2, 3}, {1, 5, 2}});
        DoubleMatrix theta = new DoubleMatrix(new double[]{0, 0, 0});
        DoubleMatrix result = lg.h(theta, x);
        for (int i = 0; i < 3; i++) {
            assertThat(0.5, is(result.get(i)));
        }
    }

    @Test
    public void test_h_while_theta_equals_to_one() {
        DoubleMatrix x = new DoubleMatrix(new double[][]{{1, 1, 2}, {1, 2, 3}});
        DoubleMatrix theta = new DoubleMatrix(new double[]{1, 1, 1});
        DoubleMatrix result = lg.h(theta, x);
        assertThat(0.9820137900379085, is(result.get(0)));
        assertThat(0.9975273768433653, is(result.get(1)));
    }

    @Test
    public void test_update_theta() {
        DoubleMatrix x = new DoubleMatrix(new double[][]{{1, 1, 2}, {1, 2, 3}});
        DoubleMatrix theta = new DoubleMatrix(new double[]{1, 1, 1});
        DoubleMatrix y = new DoubleMatrix(new double[]{1, 0});
        DoubleMatrix result = lg.updateTheta(theta, x, y);
        assertThat(0.004897705834406369, is(result.get(0)));
        assertThat(0.009885342718623196, is(result.get(1)));
        assertThat(0.014783048553029565, is(result.get(2)));
    }

    @Test
    public void test_regularized_update_theta() {
        DoubleMatrix x = new DoubleMatrix(new double[][]{{1, 1, 2}, {1, 2, 3}});
        DoubleMatrix theta = new DoubleMatrix(new double[]{1, 1, 1});
        DoubleMatrix y = new DoubleMatrix(new double[]{1, 0});
        DoubleMatrix result = lg.regularizedUpdateTheta(theta, x, y);
        assertThat(0.004897705834406369, is((result.get(0))));
        assertThat(-0.010114657281376805, is(result.get(1)));
        assertThat(-0.0052169514469704355, is(result.get(2)));
    }

    @Test
    public void test_cost_function() {
        DoubleMatrix x = new DoubleMatrix(new double[][]{{1, 1, 2}, {1, 2, 3}});
        DoubleMatrix theta = new DoubleMatrix(new double[]{1, 1, 1});
        DoubleMatrix y = new DoubleMatrix(new double[]{1, 0});
        double result = lg.costFunction(theta, x, y);
        assertThat(3.0103128065277938, is(result));
    }

    @Test
    public void test_regularized_cost_function() {
        DoubleMatrix x = new DoubleMatrix(new double[][]{{1, 1, 2}, {1, 2, 3}});
        DoubleMatrix theta = new DoubleMatrix(new double[]{1, 1, 1});
        DoubleMatrix y = new DoubleMatrix(new double[]{1, 0});
        double result = lg.regularizedCostFunction(theta, x, y);
        assertThat(5.010312806527794, is(result));
    }

    @Test
    public void test_h_to_y() {
        DoubleMatrix h = new DoubleMatrix(new double[]{0.98, 0.5, 0.3, 0.6, 0.1});
        int[] actual = {1, 1, 0, 1, 0};
        DoubleMatrix result = lg.hToY(h.transpose());
        assertThat(actual, is(result.toIntArray()));
    }

    @Test
    public void test_types_confusion_matrix() {
        DoubleMatrix actual = new DoubleMatrix(new double[]{1, 1, 1, 0, 0, 1, 1, 0, 0});
        DoubleMatrix predicted = new DoubleMatrix(new double[]{1, 0, 1, 1, 0, 1, 1, 1, 0});
        int[] expected = {2, 2, 1, 4};
        DoubleMatrix result = lg.typesConfusionMatrixes(predicted.transpose(), actual.transpose());
        assertThat(expected, is(result.toIntArray()));
    }

    @Test
    public void test_confusion_matrix() {
        String[] predicted = {"LongEaredOwl", "SnowyOwl", "LongEaredOwl", "SnowyOwl", "BarnOwl"};
        DoubleMatrix actual = new DoubleMatrix(new double[][]{{1, 1, 0}, {1, 0, 0}, {0, 1, 1}});
        DoubleMatrix result = lg.confusionMatrix(predicted, owls);
        assertThat(actual, is(result));
    }

    @Test
    public void test_get_precision() {
        DoubleMatrix confusionMat = new DoubleMatrix(new double[][]{{1, 1, 0}, {1, 0, 0}, {0, 1, 1}});
        double actual = 0.5;
        double result = lg.getPrecision(confusionMat);
        assertThat(actual, is(result));
    }

    @Test
    public void test_get_accuracy() {
        DoubleMatrix confusionMat = new DoubleMatrix(new double[][]{{1, 1, 0}, {1, 0, 0}, {0, 1, 1}});
        double actual = 0.6;
        double result = lg.getAccuracy(confusionMat);
        assertThat(actual, is(result));
    }

    @Test
    public void test_get_fp_rate() {
        DoubleMatrix confusionMat = new DoubleMatrix(new double[][]{{1, 1, 0}, {1, 0, 0}, {0, 1, 1}});
        double actual = 0.5;
        double result = lg.getFPRate(confusionMat);
        assertThat(actual, is(result));
    }

    @Test
    public void test_get_fn_rate() {
        DoubleMatrix confusionMat = new DoubleMatrix(new double[][]{{1, 1, 0}, {1, 0, 0}, {0, 1, 1}});
        double actual = 0.6666666666666666;
        double result = lg.getFNRate(confusionMat);
        assertThat(actual, is(result));
    }

    @Test
    public void test_get_recall() {
        DoubleMatrix confusionMat = new DoubleMatrix(new double[][]{{1, 1, 0}, {1, 0, 0}, {0, 1, 1}});
        double actual = 1.0 / 3.0;
        double result = lg.getRecall(confusionMat);
        assertThat(actual, is(result));
    }

    @Test
    public void test_get_f1_score() {
        DoubleMatrix confusionMat = new DoubleMatrix(new double[][]{{1, 1, 0}, {1, 0, 0}, {0, 1, 1}});
        double actual = 0.4;
        double result = lg.getF1Score(confusionMat);
        assertThat(actual, is(result));
    }

    @Test
    public void test_binaryY() {
        DoubleMatrix result = lg.binaryY("LongEaredOwl", owls);
        int[] actual = {1, 1, 0, 0, 0};
        assertThat(actual, is(result.toIntArray()));
    }

    @Test
    public void test_normalized() {
        DoubleMatrix x = new DoubleMatrix(new double[][]{{1.1, 5.3, 2.6}, {7.5, 2.8, 9.1}});
        DoubleMatrix result = lg.normalized(x);
        DoubleMatrix actual = new DoubleMatrix(new double[][]{{0, 1, 0}, {1, 0, 1}});
        assertThat(actual, is(result));
    }

    @Test
    public void test_final_output_labels() {
        DoubleMatrix h = new DoubleMatrix(new double[][]{{0.9, 0.3, 0.1}, {0.8, 0.9, 0.1}, {0.5, 0.3, 0.4}, {0.4, 0.6, 0.7}});
        String[] actual = {"LongEaredOwl", "SnowyOwl", "LongEaredOwl", "BarnOwl"};
        String[] result = lg.finalOutputLabels(h);
        assertThat(actual, is(result));
    }

    @After
    public void tearDown() throws AssignmentException {
    }

}
