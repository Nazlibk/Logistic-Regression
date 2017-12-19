package nazli.assignment3;

import nazli.assignment3.util.FileUtility;
import nazli.assignment3.util.OutputUtil;

import java.util.List;

public class Main {

    final public static int MAXIMUM_RUN_NUMBER = 10;

    public static void main(String... args) throws AssignmentException {

        FileUtility fileUtility = new FileUtility();
        OutputUtil outputUtil = new OutputUtil();
        String[] columnsHeader = {"Body Length", "Wing Length", "Body Width", "Wing Width", "Type"};
        List<Owl> owls = fileUtility.readFromCsvFile("owls15", columnsHeader);
        LogisticRegression logisticRegression = new LogisticRegression(owls);
        LogisticRegression[] logisticRegressionsResults = new LogisticRegression[MAXIMUM_RUN_NUMBER];
        for (int i = 0; i < MAXIMUM_RUN_NUMBER; i++) {
            logisticRegressionsResults[i] = logisticRegression.run();
        }
        LogisticRegression results = outputUtil.averageOfResults(logisticRegressionsResults);
        outputUtil.printResults(results);
    }
}
