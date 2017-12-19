package nazli.assignment3.util;

import nazli.assignment3.AssignmentException;
import nazli.assignment3.Owl;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FileUtility {

    /**
     * This method writes the predicted and actual labels on Labels.csv file.
     *
     * @param predictedLabels
     * @param testSet
     * @throws AssignmentException
     */
    public void writeOnCsvFile(String[] predictedLabels, List<Owl> testSet) throws AssignmentException {

        String fileName = "Labels";
        String[] columHeaders = {"Real Labels", "Predicted Labels"};
        FileWriter fileWriter = null;
        CSVPrinter csvFilePrinter = null;
        CSVFormat csvFileFormat = CSVFormat.DEFAULT.withRecordSeparator("\n");
        try {

            fileWriter = new FileWriter(fileName + ".csv");
            csvFilePrinter = new CSVPrinter(fileWriter, csvFileFormat);
            csvFilePrinter.printRecord(columHeaders);
            for (int i = 0; i < predictedLabels.length; i++) {
                List<String> labelsRecords = new ArrayList<>();
                labelsRecords.add(testSet.get(i).getType());
                labelsRecords.add(predictedLabels[i]);
                csvFilePrinter.printRecord(labelsRecords);
            }
        } catch (Exception e) {
            throw new AssignmentException("Couldn't open " + fileName + ".", e);
        } finally {
            try {
                fileWriter.flush();
                fileWriter.close();
                csvFilePrinter.close();
            } catch (IOException e) {
                throw new AssignmentException("Error while flushing/closing fileWriter/csvPrinter!!!", e);
            }
        }
    }

    /**
     * This file read data from file.
     *
     * @param fileName
     * @param columnsHeader
     * @return a list of owls
     * @throws AssignmentException
     */
    public List<Owl> readFromCsvFile(String fileName, String[] columnsHeader) throws AssignmentException {

        List<Owl> owls = new ArrayList<>();
        FileReader fileReader = null;
        CSVParser csvFileParser = null;
        CSVFormat csvFileFormat = CSVFormat.DEFAULT.withHeader(columnsHeader);
        try {
            fileReader = new FileReader(fileName + ".csv");
            csvFileParser = new CSVParser(fileReader, csvFileFormat);
            List<CSVRecord> csvRecords = csvFileParser.getRecords();
            for (int i = 0; i < csvRecords.size(); i++) {
                CSVRecord record = csvRecords.get(i);
                Owl owl = new Owl.Builder()
                        .withBodyLength(Double.parseDouble(record.get("Body Length")))
                        .withWingLength(Double.parseDouble(record.get("Wing Length")))
                        .withBodyWidth(Double.parseDouble(record.get("Body Width")))
                        .withWingWidth(Double.parseDouble(record.get("Wing Width")))
                        .withType(record.get("Type")).build();
                owls.add(owl);
            }
        } catch (Exception e) {
            throw new AssignmentException("Couldn't read from file " + fileName + ".", e);
        } finally {
            try {
                fileReader.close();
                csvFileParser.close();
            } catch (IOException e) {
                throw new AssignmentException("Error while closing fileReader/csvFileParser !!!", e);
            }
        }
        return owls;
    }
}
