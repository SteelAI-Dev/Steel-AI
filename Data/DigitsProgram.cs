using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace Steel_AI.Data
{
    internal static class DigitsProgram
    {
        public static void Run(string[] args)
        {
            int epochs = 20;
            int hidden = 64;
            double learnRate = 0.05;
            if (args.Length > 1) int.TryParse(args[1], out epochs);
            if (args.Length > 2) int.TryParse(args[2], out hidden);
            if (args.Length > 3) double.TryParse(args[3], NumberStyles.Float, CultureInfo.InvariantCulture, out learnRate);
            Console.WriteLine("===========================================");
            Console.WriteLine(" STEEL AI: Handwritten Digits (Custom CSV)");
            Console.WriteLine("===========================================\n");
            // 1\. Locate dataset file
            string customPath = @"Data\my_custom_digits.csv";
            string mnistPath = @"Data\mnist_train.csv";
            string? csvPath = File.Exists(customPath) ? customPath : (File.Exists(mnistPath) ? mnistPath : null);
            // If no dataset exists, stop early before loading variables
            if (csvPath == null)
            {
                Console.WriteLine("[ERROR] Dataset file not found!");
                Console.WriteLine($" -> Launch CanvasForm and click 'Save Sample' to create '{customPath}'.");
                Console.WriteLine($" -> Or place '{mnistPath}' inside your Data folder.");
                Console.WriteLine("\nPress any key to exit..."); Console.ReadKey();
                return;
            }
            // 2\. Declare dataset variables in the main method scope
            Console.WriteLine($"[INFO] Loading dataset from '{csvPath}'...");
            var (inputs, targets, labels) = LoadCsvDataset(csvPath);
            Console.WriteLine($"Training samples: {inputs.Length}");
            Console.WriteLine($"Network architecture: 784 inputs ->; {hidden} hidden -> 10 outputs");
            Console.WriteLine($"Learning rate: {learnRate}, Epochs: {epochs}\n");
            var network = new NeuralNetwork(784, hidden, 10);
            var algorithm = new Algorithm();
            var rng = new Random(25);
            int[] order = [.. Enumerable.Range(0, inputs.Length)];
            var totalTimer = Stopwatch.StartNew();
            Console.WriteLine($"Before training accuracy: {Accuracy(network, algorithm, inputs, labels):P2}\n");
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                var epochTimer = Stopwatch.StartNew();
                // Shuffle dataset each epoch
                for (int i = order.Length - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }
                foreach (int idx in order)
                {
                    Algorithm.ForwardPropagation(network.Layers, inputs[idx]);
                    Algorithm.CalculateGradients(network.Layers, targets[idx]);
                    Algorithm.UpdateParameters(network.Layers, learnRate);
                }
                double currentAccuracy = Accuracy(network, algorithm, inputs, labels);
                Console.WriteLine($"Epoch {epoch,2}/{epochs} - Accuracy {currentAccuracy:P2} - {epochTimer.Elapsed.TotalSeconds:F1}s");
            }
            Console.WriteLine($"\nTotal training time: {totalTimer.Elapsed.TotalSeconds:F1}s");
            PrintPerDigitAccuracy(network, algorithm, inputs, labels);
            ShowSamples(network, algorithm, inputs, labels, Math.Min(3, inputs.Length));
            // Save weights for CanvasForm real-time predictions
            NetworkState.SaveNetworkState(network, "digit_model.json");
            Console.WriteLine("\nModel saved to 'digit_model.json'.");
        }
        /// <summary> 
        /// Safely reads CSV dataset formatted as: label, pixel0, pixel1, ..., pixel783 
        /// </summary> 
        private static (double[][] inputs, double[][] targets, int[] labels) LoadCsvDataset(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"CSV dataset file not found: {Path.GetFullPath(path)}");
            }
            var lines = File.ReadAllLines(path)
                .Where(l => !string.IsNullOrWhiteSpace(l))
                .ToList(); if (lines.Count == 0)
            {
                throw new InvalidOperationException($"CSV dataset file '{path}' is empty.");
            }
            // Skip header row if present
            if (lines[0].StartsWith("label", StringComparison.OrdinalIgnoreCase))
            {
                lines.RemoveAt(0);
            }
            var inputsList = new List<double[]>();
            var targetsList = new List<double[]>();
            var labelsList = new List<int>();

            for (int i = 0; i < lines.Count; i++)
            {
                var tokens = lines[i].Split(',');

                // 1. Must have at least 785 elements (1 label + 784 pixels)
                if (tokens.Length < 785) continue;

                if (!int.TryParse(tokens[0], out int label) || label < 0 || label > 9)
                {
                    continue;
                }

                // 2. Build target vector
                double[] target = new double[10];
                target[label] = 1.0;

                // 3. Extract 784 normalized pixel inputs safely
                double[] input = new double[784];
                bool validRow = true;

                for (int p = 0; p < 784; p++)
                {
                    if (double.TryParse(tokens[p + 1], CultureInfo.InvariantCulture, out double val))
                    {
                        input[p] = val > 1.0 ? val / 255.0 : val;
                    }
                    else
                    {
                        validRow = false;
                        break;
                    }
                }

                if (validRow)
                {
                    inputsList.Add(input);
                    targetsList.Add(target);
                    labelsList.Add(label);
                }
            }

            if (inputsList.Count == 0)
            {
                throw new InvalidOperationException("No valid 784-pixel drawing rows were found in the CSV file.");
            }

            return (inputsList.ToArray(), targetsList.ToArray(), labelsList.ToArray());
        }
        private static int Predict(NeuralNetwork network, Algorithm algorithm, double[] input)
        {
            double[] output = Algorithm.ForwardPropagation(network.Layers, input);
            int best = 0; for (int i = 1; i < output.Length; i++)
            {
                if (output[i] > output[best]) best = i;
            }
            return best;
        }
        private static double Accuracy(NeuralNetwork network, Algorithm algorithm, double[][] inputs, int[] labels)
        {
            int correct = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                if (Predict(network, algorithm, inputs[i]) == labels[i])
                {
                    correct++;
                }
            }
            return (double)correct / inputs.Length;
        }

        private static void PrintPerDigitAccuracy(NeuralNetwork network, Algorithm algorithm, double[][] inputs, int[] labels)
        {
            var correct = new int[10];
            var total = new int[10];

            for (int i = 0; i < inputs.Length; i++)
            {
                int actual = labels[i];
                total[actual]++;
                if (Predict(network, algorithm, inputs[i]) == actual)
                {
                    correct[actual]++;
                }
            }

            Console.WriteLine("\nAccuracy by digit on dataset:");
            for (int digit = 0; digit < 10; digit++)
            {
                double accuracy = total[digit] > 0 ? (double)correct[digit] / total[digit] : 0;
                Console.WriteLine($" {digit}: {accuracy:P1} ({correct[digit]}/{total[digit]})");
            }
        }

        private static void ShowSamples(NeuralNetwork network, Algorithm algorithm, double[][] inputs, int[] labels, int count)
        {
            Console.WriteLine("\nSample test drawings as ASCII art:\n");
            for (int sample = 0; sample < count; sample++)
            {
                double[] image = inputs[sample];
                for (int row = 0; row < 28; row += 2)
                {
                    var line = new StringBuilder();
                    for (int col = 0; col < 28; col++)
                    {
                        line.Append(image[row * 28 + col] > 0.5 ? '*' : ' ');
                    }
                    Console.WriteLine(line);
                }
                Console.WriteLine();
            }
        }
    }
}