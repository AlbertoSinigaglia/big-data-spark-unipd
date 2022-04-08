import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class G072HW1 {
    static class ProductPopularityPairComparator implements Comparator<Tuple2<String, Integer>>, Serializable {
        public int compare(Tuple2<String, Integer> t1, Tuple2<String, Integer> t2) {
            return t1._2.compareTo(t2._2);
        }

        @java.lang.Override
        public boolean equals(java.lang.Object obj) {
            return false;
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

        final SparkConf conf = new SparkConf(true).setAppName("G072HW1").set("spark.master", "local");
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

        final org.apache.spark.api.java.function.Function<Map<String, Integer>, Iterator<Tuple2<String, Integer>>> mapToIterable = map -> map.entrySet().stream()
            .map(entry -> new Tuple2<>(entry.getKey(), entry.getValue()))
            .iterator();

        final JavaRDD<Tuple2<String, Integer>> productCustomer = rawData
            // "explode" the lines
            .map( line -> line.split(","))
            // filter out the records with quantity <= 0
            .filter(record -> Integer.parseInt(record[3]) > 0)
            // consider only the ones from S or everything if S == "all"
            .filter(record -> Objects.equals(S, "all") || record[7].equals(S))
            // map to (P,C)
            .map(record -> new Tuple2<>(record[1], Integer.parseInt(record[6])))
            // group by (P,C) to remove duplicates
            .groupBy(t -> t)
            // consider only the first one for each group
            .map(Tuple2::_1);


        final JavaPairRDD<String, Integer> productPopularity1 = productCustomer
            //from partitions to (P, |P in partitions|)
            .mapPartitionsToPair(group -> {
                Map<String, Integer> map = new HashMap<>();
                while(group.hasNext()){
                    Tuple2<String, Integer> next = group.next();
                    map.put(next._1, map.getOrDefault(next._1, 0) + 1);
                }
                return mapToIterable.call(map);
            })
            // group by key P, at most K pairs per group
            .groupByKey()
            // sum relative frequency, from (P, |P in partition|) to (P, Sum |P in partition|)
            .mapPartitionsToPair(group -> {
                Map<String, Integer> map = new HashMap<>();
                while(group.hasNext()){
                    Tuple2<String, Iterable<Integer>> current = group.next();
                    map.put(current._1, StreamSupport.stream(current._2.spliterator(), false).reduce(0, Integer::sum));
                }
                return mapToIterable.call(map);
            });

        final JavaPairRDD<String, Integer> productPopularity2 = productCustomer
            // from (P,C) to (rand(K), P) where K=sqrt(N)
            .map(Tuple2::_1)
            // group by rand(K)
            .groupBy(el -> randomGenerator.nextInt(K))
            // foreach (rand(K), (P1, P2, ...))
            .map(group -> group._2.iterator())
            // move from iterator of the value to a stream for convenience
            .map(iterator -> StreamSupport.stream(Spliterators.spliteratorUnknownSize(iterator, 0), false))
            // foreach stream, generate a map of (P, |P in stream(partition)|) and then from map to iterable of tuple
            .map(stream -> stream.collect(Collectors.groupingBy(Function.identity(), Collectors.summingInt(x -> 1))))
            .flatMapToPair(mapToIterable::call)
            // same as group by ProductID and sum all partial counts
            .reduceByKey(Integer::sum);

        if(H == 0){
            final List<Tuple2<String, Integer>> pairs1 = productPopularity1.sortByKey().collect();
            final List<Tuple2<String, Integer>> pairs2 = productPopularity2.sortByKey().collect();
            System.out.println("Number of rows = " + rawData.count());
            System.out.println("Product-Customer Pairs = " + productCustomer.count());
            System.out.println("productPopularity1:");
            System.out.println(pairs1.stream().map(t -> "Product: "+t._1+" Popularity: "+t._2+"; ").collect(Collectors.joining("")));
            System.out.println("productPopularity2:");
            System.out.println(pairs2.stream().map(t -> "Product: "+t._1+" Popularity: "+t._2+"; ").collect(Collectors.joining("")));
        } else {
            final List<Tuple2<String, Integer>> pairs1 = productPopularity1.takeOrdered(H, new ProductPopularityPairComparator().reversed());
            System.out.println("Number of rows = " + rawData.count());
            System.out.println("Product-Customer Pairs = " + productCustomer.count());
            System.out.println("Top 5 Products and their Popularities");
            System.out.println(pairs1.stream().map(t -> "Product "+t._1+" Popularity "+t._2+"; ").collect(Collectors.joining("")));
        }

    }
}
