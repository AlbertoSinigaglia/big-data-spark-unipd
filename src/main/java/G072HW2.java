import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.collection.immutable.Stream;

public class G072HW2 {
    static public class Pair{
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

            Pair pair = (Pair) o;

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
