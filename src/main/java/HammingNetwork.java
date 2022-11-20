import java.lang.reflect.Array;
import java.util.Arrays;

public class HammingNetwork {

    double[][] firstLayerWeights, secondLayerWeights;

    int neuronNum, inputNum;

    public HammingNetwork(int neuronNum, int inputNum) {
        this.neuronNum = neuronNum;
        this.inputNum = inputNum;
    }

    public void learn(int[][] dataSet) {
        initFirstLayerWeights(dataSet);
        initSecondLayerWeights();
    }

    public double[] recognize(int[] noiseVector) {
        double[] currState = calcFirstState(noiseVector), prevState;

        do {
            prevState = currState.clone();

            for (int index1 = 0; index1 < secondLayerWeights.length; ++index1) {
                if (secondLayerWeights[index1].length != prevState.length) throw new RuntimeException("Length error");

                double potential = 0;

                for (int index2 = 0; index2 < secondLayerWeights[index1].length; ++index2) {
                    potential += secondLayerWeights[index1][index2] * prevState[index2];
                }

                currState[index1] = activate(potential);
            }
        } while (isStatesChanges(currState, prevState));

        return currState;
    }

    public boolean isStatesChanges(double[] currentState, double[] prevState) {
        if (currentState.length != prevState.length) throw new RuntimeException("Different states lengths");

        for (int index = 0; index < currentState.length; ++index) {
            if (currentState[index] != prevState[index]) {
                return true;
            }
        }

        return false;
    }

    private double[] calcFirstState(int[] noiseVector) {
        double[] firstState = new double[neuronNum];

        for (int index1 = 0; index1 < firstLayerWeights.length; ++index1) {
            if (firstLayerWeights[index1].length != noiseVector.length)
                throw new RuntimeException("Wrong noise vector size");

            double potential = 0;

            for (int index2 = 0; index2 < firstLayerWeights[index1].length; ++index2) {
                potential += noiseVector[index2] * firstLayerWeights[index1][index2];
            }

            firstState[index1] = activate(potential);
        }

        return firstState;
    }

    private void initFirstLayerWeights(int[][] refPatternMatrix) {
        firstLayerWeights = new double[neuronNum][inputNum];

        if (firstLayerWeights.length != refPatternMatrix.length) throw new RuntimeException("Wrong reference pattern matrix size");

        for (int index1 = 0; index1 < firstLayerWeights.length; ++index1) {
            if (firstLayerWeights[index1].length != refPatternMatrix[index1].length)
                throw new RuntimeException("Wrong reference pattern matrix size");

            for (int index2 = 0; index2 < firstLayerWeights[index1].length; ++index2) {
                firstLayerWeights[index1][index2] = (double) refPatternMatrix[index1][index2] / 2;
            }
        }
    }

    private void initSecondLayerWeights() {
        secondLayerWeights = new double[neuronNum][neuronNum];

        for (int index1 = 0; index1 < secondLayerWeights.length; ++index1) {
            for (int index2 = 0; index2 < secondLayerWeights[index1].length; ++index2) {
                if (index1 != index2) {
                    secondLayerWeights[index1][index2] = -1. / neuronNum;
                }
                else {
                    secondLayerWeights[index1][index2] = 1;
                }
            }
        }
    }

    private double activate(double potential) {
        double tNum = (double) inputNum / 2;

        if (potential > tNum) return tNum;

        return (potential > 0) ? potential : 0;
    }

    public static void main(String[] args) {
        int[][] dataSet = {
                {1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1},
                {-1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1},
                {1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1},
                {1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1},
                {1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1},
                {1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1},
                {1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1},
                {1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1},
                {1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1},
                {1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1}
        };

        int[] noiseVector = {1, 1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1};

        HammingNetwork hammingNetwork = new HammingNetwork(10, 15);
        hammingNetwork.learn(dataSet);
        System.out.println(Arrays.toString(hammingNetwork.recognize(noiseVector)));

        for (int index1 = 0; index1 < hammingNetwork.firstLayerWeights.length; ++index1) {
            for (int index2 = 0; index2 < hammingNetwork.firstLayerWeights[index1].length; ++index2) {
                System.out.print(hammingNetwork.firstLayerWeights[index1][index2] + " ");
            }
            System.out.println();
        }

        System.out.println();

        for (int index1 = 0; index1 < hammingNetwork.secondLayerWeights.length; ++index1) {
            for (int index2 = 0; index2 < hammingNetwork.secondLayerWeights[index1].length; ++index2) {
                System.out.print(hammingNetwork.secondLayerWeights[index1][index2] + " ");
            }
            System.out.println();
        }

    }

}
