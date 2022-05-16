import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 *
 *
 *
 *
 *
 *       GHE ZE UN BUG QUA,
 *       MA TANTO E' APPURATO
 *       CHE LA CACHE NON
 *       AIUTA AFFATTO
 *
 *
 *
 *
 *
 */

public class G072HW2UsingCache {
    static public class VectorWeightPair {
        public Vector vec;
        public Long weight;
        public Integer index;

        @Override
        public String toString() {
            return "Pair{" +
                "vec=" + vec +
                ", weight=" + weight +
                '}';
        }

        public VectorWeightPair(Vector vec, Long weight, Integer index) {
            this.vec = vec;
            this.weight = weight;
            this.index = index;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            VectorWeightPair that = (VectorWeightPair) o;

            if (!Objects.equals(vec, that.vec)) return false;
            if (!Objects.equals(weight, that.weight)) return false;
            return Objects.equals(index, that.index);
        }

        @Override
        public int hashCode() {
            int result = vec != null ? vec.hashCode() : 0;
            result = 31 * result + (weight != null ? weight.hashCode() : 0);
            result = 31 * result + (index != null ? index.hashCode() : 0);
            return result;
        }
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
            .map(G072HW2::strToVector)
            .forEach(result::add);
        return result;
    }


    static void initCache(int N, ArrayList<VectorWeightPair> vec){
        distances = new Double[N][N];
        for(int i=0; i < N; i++){
            for(int j=i; j < N; j++){
                distances[vec.get(i).index][vec.get(j).index] = distances[vec.get(j).index][vec.get(i).index] = Math.sqrt(Vectors.sqdist(vec.get(i).vec, vec.get(j).vec));
            }
        }
    }
    static Double[][] distances = null;
    static double getDistance(VectorWeightPair v1, VectorWeightPair v2){
        return distances[v1.index][v2.index];
    }

    static ArrayList<Vector> SeqWeightedOutliers(final ArrayList<Vector> P, final ArrayList<Long> W, final int k, final int z, final float alpha){
        double r = Double.POSITIVE_INFINITY;
        ArrayList<VectorWeightPair> pairs = new ArrayList<>();
        for(int i = 0 ; i < W.size(); i++){
            pairs.add(new VectorWeightPair(P.get(i), W.get(i), i));
        }
        initCache(pairs.size(), pairs);
        for(int i = 0; i <  k + z + 1; i++) {
            for (int j = i + 1; j < k + z + 1; j++) {
                r = Math.min(r, getDistance(pairs.get(i), pairs.get(j)) / 2);
            }
        }
        System.out.println("Initial guess = "+r);

        int guesses = 1;
        final long wTot = W.stream().mapToLong(l -> l).sum();

        while(true){
            long wTemp = wTot;
            ArrayList<VectorWeightPair> Z_pairs = new ArrayList<>(pairs);
            ArrayList<Vector> S = new ArrayList<>();
            while(S.size() < k && wTemp > 0){
                long max = -1;
                VectorWeightPair newCenter = null;
                for(VectorWeightPair x : pairs){
                    long ballWeight = 0;
                    for (VectorWeightPair other : Z_pairs) {
                        if (getDistance(other, x) <= (1 + 2 * alpha) * r) {
                            ballWeight += other.weight;
                        }
                    }
                    if(ballWeight > max){
                        max = ballWeight;
                        newCenter = x;
                    }
                }
                S.add(newCenter.vec);
                for(int i = 0; i < Z_pairs.size(); i++) {
                    if (getDistance(Z_pairs.get(i), newCenter) <= (3 + 4 * alpha) * r) {
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

    static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z){
        ArrayList<Double> dist = new ArrayList<>();
        for(Vector v1: P){
            double min = Double.POSITIVE_INFINITY;
            for(Vector v2: S){
                min = Math.min(min, Math.sqrt(Vectors.sqdist(v1, v2)));
            }
            dist.add(min);
        }
        Collections.sort(dist);

        return dist.get(dist.size() - 1 - z);
    }



    public static void main(String[] args) throws Exception {
        final String path = args[0];
        final int K = Integer.parseInt(args[1]);
        final int Z = Integer.parseInt(args[2]);
        final ArrayList<Vector> inputPoints =  readVectorsSeq(path);
        final ArrayList<Long> weights = new ArrayList<>(Collections.nCopies(inputPoints.size(), 1L));
        System.out.println("Input size n = "+inputPoints.size());
        System.out.println("Number of centers k = "+K);
        System.out.println("Number of outliers z = "+Z);
        long startTime = System.currentTimeMillis();
        final ArrayList<Vector> solution = SeqWeightedOutliers(inputPoints, weights, K, Z , 0);
        long endTime = System.currentTimeMillis();
        final double objective = ComputeObjective(inputPoints, solution, Z);
        System.out.println("Objective function = "+objective);
        System.out.println("Time of SeqWeightedOutliers = "+(endTime - startTime));
    }

}
