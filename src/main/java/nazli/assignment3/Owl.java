package nazli.assignment3;

public class Owl {

    private double bodyLength;
    private double wingLength;
    private double bodyWidth;
    private double wingWidth;
    private String type;

    public Owl(double bodyLength, double wingLength, double bodyWidth, double wingWidth, String type) {
        this.bodyLength = bodyLength;
        this.wingLength = wingLength;
        this.bodyWidth = bodyWidth;
        this.wingWidth = wingWidth;
        this.type = type;
    }

    public Owl() {
    }

    public double getBodyLength() {
        return bodyLength;
    }

    public void setBodyLength(double bodyLength) {
        this.bodyLength = bodyLength;
    }

    public double getWingLength() {
        return wingLength;
    }

    public void setWingLength(double wingLength) {
        this.wingLength = wingLength;
    }

    public double getBodyWidth() {
        return bodyWidth;
    }

    public void setBodyWidth(double bodyWidth) {
        this.bodyWidth = bodyWidth;
    }

    public double getWingWidth() {
        return wingWidth;
    }

    public void setWingWidth(double wingWidth) {
        this.wingWidth = wingWidth;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    @Override
    public String toString() {
        return "nazli.assignment3.Owl{" +
                "bodyLength=" + bodyLength +
                ", wingLength=" + wingLength +
                ", bodyWidth=" + bodyWidth +
                ", wingWidth=" + wingWidth +
                ", type='" + type + '\'' +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Owl owl = (Owl) o;

        if (Double.compare(owl.bodyLength, bodyLength) != 0) return false;
        if (Double.compare(owl.wingLength, wingLength) != 0) return false;
        if (Double.compare(owl.bodyWidth, bodyWidth) != 0) return false;
        if (Double.compare(owl.wingWidth, wingWidth) != 0) return false;
        return type != null ? type.equals(owl.type) : owl.type == null;
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        temp = Double.doubleToLongBits(bodyLength);
        result = (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(wingLength);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(bodyWidth);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(wingWidth);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (type != null ? type.hashCode() : 0);
        return result;
    }

    public static class Builder {
        private double bodyLength;
        private double wingLength;
        private double bodyWidth;
        private double wingWidth;
        private String type;

        public Builder withBodyLength(double bodyLength) {
            this.bodyLength = bodyLength;
            return this;
        }

        public Builder withWingLength(double wingLength) {
            this.wingLength = wingLength;
            return this;
        }

        public Builder withBodyWidth(double bodyWidth) {
            this.bodyWidth = bodyWidth;
            return this;
        }

        public Builder withWingWidth(double wingWidth) {
            this.wingWidth = wingWidth;
            return this;
        }

        public Builder withType(String type) {
            this.type = type;
            return this;
        }

        public Owl build() {
            return new Owl(bodyLength, wingLength, bodyWidth, wingWidth, type);
        }
    }
}

