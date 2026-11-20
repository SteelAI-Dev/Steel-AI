using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Steel_AI.NetworkDefinition;

namespace Steel_AI.NetworkAlgorithm
{
    internal class Trainer
    {
        private readonly Algorithm _algorithm = new();

        public void Train(NeuralNetwork network, double[][] trainingData, double[][] expectedOutputs, int epochs, double learningRate)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalError = 0;
                for (int i = 0; i < trainingData.Length; i++)
                {
                    double[] predictions = _algorithm.ForwardPropagation(network.Layers, trainingData[i]);
                    double error = expectedOutputs[i][0] - predictions[0];
                    totalError += error * error;
                    _algorithm.CalculateGradients(network.Layers, expectedOutputs[i]);
                    _algorithm.UpdateParameters(network.Layers, learningRate);
                }
                if ((epoch + 1) % 4000 == 0)
                {
                    Console.WriteLine($"Epoch {epoch + 1}/{epochs} - Loss: {totalError:F6}");
                }
            }
        }
    }
}
