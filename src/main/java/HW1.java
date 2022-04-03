import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import java.io.Serializable;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class HW1 {
    static class ProductPopularityPairComparator implements Comparator<Tuple2<String, Long>>, Serializable {
        public int compare(Tuple2<String, Long> t1, Tuple2<String, Long> t2) {
            return t1._2.compareTo(t2._2);
        }
    }

    public static void main(String[] args) {
        /*
            To run use:
                4  0  Italy           datasets/hw1/sample_50.csv
                4  5  all             datasets/hw1/sample_10000.csv
                4  5  United_Kingdom  datasets/hw1/full_dataset.csv
         */
        if (args.length < 4) {
            throw new IllegalArgumentException("USAGE: num_partitions top_products country_name_filter file_path");
        }

        final SparkConf conf = new SparkConf(true).setAppName("WordCount").set("spark.master", "local");
        final JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        // Read number of partitions
        final int K = Integer.parseInt(args[0]);
        // Read top n-elements
        final int H = Integer.parseInt(args[1]);
        // Read country filter
        final String S = args[2];
        // Read input file and subdivide it into K random partitions
        final JavaRDD<String> rawData = sc.textFile(args[3]).repartition(K).cache();
        // Random generator for partitions
        final Random randomGenerator = new Random();

        final var productCustomer = rawData
            // "explode" the lines
            .map( line -> line.split(","))
            // filter out the records with quantity <= 0
            .filter(record -> Integer.parseInt(record[3]) > 0)
            // consider only the ones from S or everything if S == "all"
            .filter(record -> Objects.equals(S, "all") || record[7].equals(S))
            // map to (P,C)
            .map(record -> new Tuple2<>(record[1], Integer.parseInt(record[6])))
            // map to ((P,C), (P,C))
            .mapToPair(pair -> new Tuple2<>(pair, pair))
            // group pairs of (P,C), removing duplicate (P,C) pairs
            .groupByKey()
            // consider only the first one for each group
            .map(Tuple2::_1);


        final var productPopularity1 = productCustomer
            //from partitions to (P, |P in partitions|)
            .mapPartitionsToPair(group -> {
                var map = new HashMap<String, Long>();
                while(group.hasNext()){
                    var next = group.next();
                    map.put(next._1, map.getOrDefault(next._1, 0L) + 1);
                }
                return map.entrySet().stream()
                    .map(entry-> new Tuple2<>(entry.getKey(), entry.getValue()))
                    .iterator();
            })
            // group by key P, at most K pairs per group
            .groupByKey()
            // sum relative frequency, from (P, |P in partition|) to (P, Sum |P in partition|)
            .mapPartitionsToPair(group -> {
                var map = new HashMap<String, Long>();
                while(group.hasNext()){
                    var current = group.next();
                    map.put(current._1, StreamSupport.stream(current._2.spliterator(), false).reduce(0L, Long::sum));
                }
                return map.entrySet().stream()
                    .map(entry-> new Tuple2<>(entry.getKey(), entry.getValue()))
                    .iterator();
            });

        final var productPopularity2 = productCustomer
            // from (P,C) to (rand(K), P) where K=sqrt(N)
            .mapToPair( pair -> new Tuple2<>(randomGenerator.nextInt(K), pair._1))
            // group by rand(K)
            .groupByKey()
            // foreach (rand(K), (P1, P2, ...))
            .map(group -> group._2.iterator())
            // move from iterator of the value to a stream for convenience
            .map(iterator ->
                Stream.generate(() -> null)
                    .takeWhile(x -> iterator.hasNext())
                    .map(n -> iterator.next()))
            // foreach stream, generate a map of (P, |P in stream(partition)|) and then from map to list of tuple
            .flatMapToPair(stream ->
                stream.collect(Collectors.groupingBy(Function.identity(), Collectors.counting()))
                    .entrySet()
                    .stream()
                    .map(entry-> new Tuple2<>(entry.getKey(), entry.getValue()))
                    .iterator())
            // group by key P, at most K pairs per group
            .groupByKey()
            // sum relative frequency, from (P, |P in partition|) to (P, Sum |P in partition|)
            .map(tuple -> new Tuple2<>(tuple._1, StreamSupport.stream(tuple._2.spliterator(), false).reduce(0L, Long::sum)));

        final var pairs1 = H > 0 ?
            // take top H elements, using a ProductPopularityPairComparator, but reversed (top = highest)
            productPopularity1.takeOrdered(H, new ProductPopularityPairComparator().reversed()) :
            // take all elements
            productPopularity1.collect();

        final var pairs2 = H > 0 ?
            // take top H elements, using a ProductPopularityPairComparator, but reversed (top = highest)
            productPopularity2.takeOrdered(H, new ProductPopularityPairComparator().reversed()) :
            // take all elements
            productPopularity2.collect();

        System.out.println("Number of rows = " + rawData.count());
        System.out.println("Product-Customer Pairs = " + productCustomer.count());
        System.out.println("Top 5 Products and their Popularities");
        Stream.of(pairs1.stream(),pairs2.stream())
            .map(s -> s.map(t -> "Product "+t._1+" Popularity "+t._2+"; ").collect(Collectors.joining("")))
            .forEach(System.out::println);

    }
}
