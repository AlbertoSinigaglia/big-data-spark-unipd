import com.google.common.collect.Lists;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Array;
import scala.Serializable;
import scala.Tuple2;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class G072HW3 {

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// MAIN PROGRAM 
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {

        if (args.length != 4) {
            throw new IllegalArgumentException("USAGE: filepath k z L");
        }

        // ----- Initialize variables
        String filename = args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);
        int L = Integer.parseInt(args[3]);
        long start, end; // variables for time measurements

        // ----- Set Spark Configuration
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("MR k-center with outliers").set("spark.master", "local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // ----- Read points from file
        start = System.currentTimeMillis();
        JavaRDD<Vector> inputPoints = sc.textFile(args[0], L)
            .map(G072HW3::strToVector)
            .repartition(L)
            .cache();
        long N = inputPoints.count();
        end = System.currentTimeMillis();

        // ----- Pring input parameters
        System.out.println("File : " + filename);
        System.out.println("Number of points N = " + N);
        System.out.println("Number of centers k = " + k);
        System.out.println("Number of outliers z = " + z);
        System.out.println("Number of partitions L = " + L);
        System.out.println("Time to read from file: " + (end-start) + " ms");

        // ---- Solve the problem
        ArrayList<Vector> solution = MR_kCenterOutliers(inputPoints, k, z, L);

        // ---- Compute the value of the objective function
        start = System.currentTimeMillis();
        double objective = computeObjective(inputPoints, solution, z);
        end = System.currentTimeMillis();
        System.out.println("Objective function = " + objective);
        System.out.println("Time to compute objective function: " + (end-start) + " ms");

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// AUXILIARY METHODS
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method strToVector: input reading
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method euclidean: distance function
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double euclidean(Vector a, Vector b) {
        return Math.sqrt(Vectors.sqdist(a, b));
    }


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Method MR_kCenterOutliers: MR algorithm for k-center with outliers
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> MR_kCenterOutliers (JavaRDD<Vector> points, int k, int z, int L)
    {

        //------------- ROUND 1 ---------------------------
        // to test: .map(p -> new Tuple2<>(p, 1L));
        // and change alpha to 0
        JavaRDD<Tuple2<Vector,Long>> coreset = points.mapPartitions(x ->
        {
            ArrayList<Vector> partition = new ArrayList<>();
            while (x.hasNext()) partition.add(x.next());
            ArrayList<Vector> centers = kCenterFFT(partition, k+z+1);
            ArrayList<Long> weights = computeWeights(partition, centers);
            ArrayList<Tuple2<Vector,Long>> c_w = new ArrayList<>();
            for(int i =0; i < centers.size(); ++i)
            {
                Tuple2<Vector, Long> entry = new Tuple2<>(centers.get(i), weights.get(i));
                c_w.add(i,entry);
            }
            return c_w.iterator();
        }); // END OF ROUND 1

        //------------- ROUND 2 ---------------------------
        long start = System.currentTimeMillis();
        ArrayList<Tuple2<Vector, Long>> elems = new ArrayList<>((k+z)*L);
        elems.addAll(coreset.collect());
        long end = System.currentTimeMillis();
        System.out.println("Time taken by round 1: " + (end-start) + " ms");

        //
        // ****** ADD YOUR CODE
        // ****** Compute the final solution (run SeqWeightedOutliers with alpha=2)
        // ****** Measure and print times taken by Round 1 and Round 2, separately
        // ****** Return the final solution
        //

        /*JavaRDD<ArrayList<Vector>> centersRDD = coreset
            .groupBy(el -> null)
            .map(all -> {
                ArrayList<Vector> vec = new ArrayList<>();
                ArrayList<Long> weights = new ArrayList<>();
                all._2.forEach(el -> {
                    vec.add(el._1);
                    weights.add(el._2);
                });
                return SeqWeightedOutliers(
                    vec,
                    weights,
                    k, z, 2);
            });*/


        ArrayList<Vector> centers = new ArrayList<>(k);
        start = System.currentTimeMillis();
        centers.addAll(
            SeqWeightedOutliers(
                new ArrayList<>(elems.stream().map(el -> el._1).collect(Collectors.toList())),
                new ArrayList<>(elems.stream().map(el -> el._2).collect(Collectors.toList())),
                k, z, 2
            )
        );
        end = System.currentTimeMillis();
        System.out.println("Time taken by round 2: " + (end-start) + " ms");
        return centers;

    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method kCenterFFT: Farthest-First Traversal
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> kCenterFFT (ArrayList<Vector> points, int k) {

        final int n = points.size();
        double[] minDistances = new double[n];
        Arrays.fill(minDistances, Double.POSITIVE_INFINITY);

        ArrayList<Vector> centers = new ArrayList<>(k);

        Vector lastCenter = points.get(0);
        centers.add(lastCenter);
        double radius =0;

        for (int iter=1; iter<k; iter++) {
            int maxIdx = 0;
            double maxDist = 0;

            for (int i = 0; i < n; i++) {
                double d = euclidean(points.get(i), lastCenter);
                if (d < minDistances[i]) {
                    minDistances[i] = d;
                }

                if (minDistances[i] > maxDist) {
                    maxDist = minDistances[i];
                    maxIdx = i;
                }
            }

            lastCenter = points.get(maxIdx);
            centers.add(lastCenter);
        }
        return centers;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeWeights: compute weights of coreset points
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Long> computeWeights(ArrayList<Vector> points, ArrayList<Vector> centers)
    {
        Long weights[] = new Long[centers.size()];
        Arrays.fill(weights, 0L);
        for(int i = 0; i < points.size(); ++i)
        {
            double tmp = euclidean(points.get(i), centers.get(0));
            int mycenter = 0;
            for(int j = 1; j < centers.size(); ++j)
            {
                if(euclidean(points.get(i),centers.get(j)) < tmp)
                {
                    mycenter = j;
                    tmp = euclidean(points.get(i), centers.get(j));
                }
            }
            // System.out.println("Point = " + points.get(i) + " Center = " + centers.get(mycenter));
            weights[mycenter] += 1L;
        }
        ArrayList<Long> fin_weights = new ArrayList<>(Arrays.asList(weights));
        return fin_weights;
    }

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method SeqWeightedOutliers: sequential k-center with outliers
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    //
    // ****** ADD THE CODE FOR SeqWeightedOuliers from HW2
    //

    static public class Pair implements Serializable {
        public Vector vec;
        public Long weight;

        @Override
        public String toString() {
            return "Pair{" +
                "vec=" + vec +
                ", weight=" + weight +
                '}';
        }

        public Pair(Vector vec, Long weight) {
            this.vec = vec;
            this.weight = weight;
        }
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            G072HW2.Pair pair = (G072HW2.Pair) o;

            if (!Objects.equals(vec, pair.vec)) return false;
            return Objects.equals(weight, pair.weight);
        }

        @Override
        public int hashCode() {
            int result = vec != null ? vec.hashCode() : 0;
            result = 31 * result + (weight != null ? weight.hashCode() : 0);
            return result;
        }
    }
    static ArrayList<Vector> SeqWeightedOutliers(final ArrayList<Vector> P, final ArrayList<Long> W, final int k, final int z, final float alpha){
        ArrayList<Pair> pairs = new ArrayList<>();
        for(int i = 0 ; i < W.size(); i++){
            pairs.add(new Pair(P.get(i), W.get(i)));
        }
        double r = Double.POSITIVE_INFINITY;
        for(int i = 0; i <  k + z + 1; i++) {
            for (int j = i + 1; j < k + z + 1; j++) {
                r = Math.min(r, Math.sqrt(Vectors.sqdist(P.get(i), P.get(j))) / 2);
            }
        }
        System.out.println("Initial guess = "+r);

        int guesses = 1;
        long wTot = 0;
        for(long l :W){
            wTot += l;
        }

        while(true){
            long wTemp = wTot;
            ArrayList<Pair> Z_pairs = new ArrayList<>(pairs);
            ArrayList<Vector> S = new ArrayList<>();
            double rMin = (1 + 2 * alpha) * r;
            double rMax = (3 + 4 * alpha) * r;
            while(S.size() < k && wTemp > 0){
                long max = -1;
                Vector newCenter = null;
                for(Pair x : pairs){
                    long ballWeight = 0;
                    for (Pair other : Z_pairs) {
                        if (Math.sqrt(Vectors.sqdist(other.vec, x.vec)) <= rMin) {
                            ballWeight += other.weight;
                        }
                    }
                    if(ballWeight > max){
                        max = ballWeight;
                        newCenter = x.vec;
                    }
                }
                S.add(newCenter);
                for(int i = 0; i < Z_pairs.size(); i++) {
                    if (Math.sqrt(Vectors.sqdist(Z_pairs.get(i).vec, newCenter)) <= rMax) {
                        wTemp -= Z_pairs.remove(i).weight;
                        i--;
                    }
                }
            }
            if(wTemp <= z){
                System.out.println("Final guess = " + r);
                System.out.println("Number of guesses = " + guesses);
                return S;
            } else {
                guesses++;
                r = 2 * r;
            }
        }
    }



// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
// Method computeObjective: computes objective function  
// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static double computeObjective (JavaRDD<Vector> points, ArrayList<Vector> centers, int z)
    {
        /*return points.map(p -> {
            double minDist = Double.POSITIVE_INFINITY;
            for(Vector v : centers){
                double dist = euclidean(p, v);
                if(dist < minDist){
                    minDist = dist;
                }
            }
            return minDist;
        }).mapPartitionsToPair(listDistances -> {
            ArrayList<Double> dists = new ArrayList<Double>(z+1);
            while(listDistances.hasNext()){
                dists.add(listDistances.next());
            }
            Collections.sort(dists);
            return Collections.singletonList(
                new Tuple2<>(null, new ArrayList<>(dists.subList(dists.size() - 1 - z - 1, dists.size()-1)))
            ).iterator();
        })
            .groupByKey()
            .mapToDouble(listDistances -> {
                ArrayList<Double> all = new ArrayList<>();
                listDistances._2.forEach(all::addAll);
                Collections.sort(all);
                return all.get(all.size() - z );
        }).collect().get(0);*/
        return points.map(p -> {
                double minDist = Double.POSITIVE_INFINITY;
                for(Vector v : centers){
                    double dist = euclidean(p, v);
                    if(dist < minDist){
                        minDist = dist;
                    }
                }
                return minDist;
            })
            .top(z+1)
            .stream()
            .mapToDouble(d -> d)
            .min()
            .orElse(0);
    }

}