package nazli.assignment3.nazli.assihnment3.util;

import nazli.assignment3.AssignmentException;
import nazli.assignment3.Owl;
import nazli.assignment3.util.FileUtility;
import org.junit.Test;

import java.util.List;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

public class FileUtiliyTest {
    FileUtility fileUtility = new FileUtility();

    @Test
    public void test_read_from_file() throws AssignmentException {
        String[] columnsHeader = {"Body Length", "Wing Length", "Body Width", "Wing Width", "Type"};
        List<Owl> owls = fileUtility.readFromCsvFile("owls15", columnsHeader);
        Owl owl = new Owl(3, 5, 1.6, 0.2, "LongEaredOwl");
        boolean isEqual = owls.get(0).equals(owl);
        assertThat(isEqual, is(true));
    }
}
