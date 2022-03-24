import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

public class HW1 {
    public static void main(String[] args) {


        if (args.length < 4) {
            throw new IllegalArgumentException("USAGE: num_partitions top_products country_name_filter file_path");
        }

        SparkConf conf = new SparkConf(true).setAppName("WordCount").set("spark.master", "local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        // Read number of partitions
        int K = Integer.parseInt(args[0]);
        // Read top n-elements
        int H = Integer.parseInt(args[1]);
        // Read country filter
        String S = args[2];
        // Read input file and subdivide it into K random partitions
        JavaRDD<String> docs = sc.textFile(args[1]).repartition(K).cache();

        // TODO
    }
}
