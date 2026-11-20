using Steel_AI;
using Steel_AI.Data;
using System;
using System.IO;

namespace Steel_AI
{
    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            // 1. If startedd with "digits" argument: run DigitsProgram training
            if (args.Length > 0 && args[0] == "digits")
            {
                DigitsProgram.Run(args);
                return;
            }

            // 2. If started with "canvas" argument (or if you want UI by default when pressing F5):
            if (args.Length > 0 && args[0] == "canvas")
            {
                ApplicationConfiguration.Initialize();
                Application.Run(new CanvasForm());
                return;
            }
            // 3. Default Console Mode: XOR Training and Model Persistence Test	
            Console.WriteLine("===========================================");
            Console.WriteLine(" STEEL AI: Model Persistence Test (XOR) ");
            Console.WriteLine("===========================================\n");
            string modelPath = "xor_model.json";
            NeuralNetwork? network = null; Algorithm _ = new();
            // 1\. Define XOR Dataset 
            double[][] inputs =
            [
                            [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]
            ];

            double[][] targets =
            [
                            [0],
                    [1],
                    [1],
                    [0]
            ];
            // 2\. Check if Saved Model Exists
            if (File.Exists(modelPath))
            {
                Console.WriteLine($"[INFO] Found saved model at '{modelPath}'. Loading weights from disk...");
                network = NetworkState.LoadNetworkState(modelPath);
            }
            // 3\. If No Saved Model Found, Train New Network and Save 
            if (network == null)
            {
                Console.WriteLine("[INFO] No saved model found. Initializing new network and training...\n");
                network = new NeuralNetwork(2, 3, 1);
                double learningRate = 0.5;
                int epochs = 20000;
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double totalError = 0;
                    for (int i = 0; i < inputs.Length; i++)
                    {
                        double[] predictions = Algorithm.ForwardPropagation(network.Layers, inputs[i]);
                        double error = targets[i][0] - predictions[0];
                        totalError += error * error;
                        Algorithm.CalculateGradients(network.Layers, targets[i]);
                        Algorithm.UpdateParameters(network.Layers, learningRate);
                    }
                    if ((epoch + 1) % 4000 == 0)
                    {
                        Console.WriteLine($"Epoch {epoch + 1}/{epochs} - Loss: {totalError:F6}");
                    }
                }
                // Save trained network parameters 
                Console.WriteLine("\nSaving trained model state to disk...");
                NetworkState.SaveNetworkState(network, modelPath);
            }
            // 4\. Verify Model Predictions 
            Console.WriteLine("\n-------------------------------------------");
            Console.WriteLine(" MODEL PREDICTIONS VERIFICATION ");
            Console.WriteLine("-------------------------------------------");
            for (int i = 0; i < inputs.Length; i++)
            {
                double[] output = Algorithm.ForwardPropagation(network.Layers, inputs[i]);
                Console.WriteLine($"Input: [{inputs[i][0]}, {inputs[i][1]}] => Output: {output[0]:F4} | Target: {targets[i][0]}");
            }
            Console.WriteLine("\nSteel AI is a production of SteelAI Development");
            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();

        }
    }
}