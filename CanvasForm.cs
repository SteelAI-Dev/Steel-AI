#pragma warning disable
using Steel_AI.Data;
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace Steel_AI
{
    public partial class CanvasForm : Form
    {
        private Bitmap canvasBitmap;
        private Graphics canvasGraphics;
        private bool isDrawing = false;
        private PictureBox picCanvas;
        private ComboBox numDigit;
        private Button btnClear;
        private Button btnSave;
        private Button btnPredict;
        private Label lblStatus;
        private Button btnTrain;
        private Point lastPoint;

        public CanvasForm()
        {
            InitializeComponent();

            btnClear.Click += btnClear_Click;
            btnSave.Click += btnSave_Click;
            btnPredict.Click += btnPredict_Click;
			btnTrain.Click += btnTrain_Click;

			// Create 280x280 drawing bitmap
			canvasBitmap = new Bitmap(280, 280);
            canvasGraphics = Graphics.FromImage(canvasBitmap);
            canvasGraphics.Clear(Color.Black);
            canvasGraphics.SmoothingMode = SmoothingMode.AntiAlias;

            picCanvas.Image = canvasBitmap;

            // Wire mouse events
            picCanvas.MouseDown += PicCanvas_MouseDown;
            picCanvas.MouseMove += PicCanvas_MouseMove;
            picCanvas.MouseUp += (s, e) => isDrawing = false;
        }

        private void PicCanvas_MouseDown(object sender, MouseEventArgs e)
        {
            isDrawing = true;
            lastPoint = e.Location;
        }

        private void PicCanvas_MouseMove(object sender, MouseEventArgs e)
        {
            if (isDrawing)
            {
                // Use thick white pen (width ~18-20px) to mimic MNIST brush strokes
                using (Pen pen = new Pen(Color.White, 15f))
                {
                    pen.StartCap = LineCap.Round;
                    pen.EndCap = LineCap.Round;
                    canvasGraphics.DrawLine(pen, lastPoint, e.Location);
                }
                lastPoint = e.Location;
                picCanvas.Invalidate();
            }
        }

        private void btnClear_Click(object sender, EventArgs e)
        {
            canvasGraphics.Clear(Color.Black);
            picCanvas.Invalidate();
            lblStatus.Text = "Canvas cleared.";
        }

        private void btnSave_Click(object sender, EventArgs e)
        {
            int targetDigit = (int)numDigit.SelectedIndex;
            double[] pixels = GetCanvasPixelArray();
            // Create line string: "label,pixel0,pixel1,..."
            string csvRow = targetDigit + "," + string.Join(",", pixels.Select(p => (p * 255.0).ToString("F0")));
            string filePath = "Data/my_custom_digits.csv";
            // Ensure Data directory exists
            Directory.CreateDirectory("Data");
            // Append to file
            File.AppendAllLines(filePath, new[] { csvRow });
            lblStatus.Text = $"Saved digit '{targetDigit}' to {filePath}!";
            btnClear_Click(sender, e);
            // Clear canvas for next drawing
        }

		private void btnPredict_Click(object sender, EventArgs e)
		{
			string modelPath = "digit_model.json"; 
            if (!File.Exists(modelPath)) 
            { 
                lblStatus.Text = "No trained model found! Train your model first using DigitsProgram."; 
                return; 
            } 
            // Load pre-trained network weights
            NeuralNetwork network = NetworkState.LoadNetworkState(modelPath); 
            // Get 784 normalized pixel inputs from canvas
            double[] inputPixels = GetCanvasPixelArray(); 
            // Run Forward Propagation
            double[] outputActivations = Algorithm.ForwardPropagation(network.Layers, inputPixels); 
            // Find predicted digit with highest confidence (ArgMax)
            int predictedDigit = Array.IndexOf(outputActivations, outputActivations.Max()); 
            double confidence = outputActivations[predictedDigit] * 100.0; 
            lblStatus.Text = $"Prediction: Digit {predictedDigit} ({confidence:F1}% confidence)"; 
        }

		private void btnTrain_Click(object sender, EventArgs e)
		{
			lblStatus.Text = "Training neural network on CSV samples..."; 
            Application.DoEvents(); // Refresh UI label 
            // Launch DigitsProgram training directly
            DigitsProgram.Run(new string[] { "digits" }); 
            lblStatus.Text = "Training complete! Saved to digit_model.json"; 
        }

        private void InitializeComponent()
        {
            picCanvas = new PictureBox();
            numDigit = new ComboBox();
            btnClear = new Button();
            btnSave = new Button();
            btnPredict = new Button();
            lblStatus = new Label();
            btnTrain = new Button();
            ((System.ComponentModel.ISupportInitialize)picCanvas).BeginInit();
            SuspendLayout();
            // 
            // picCanvas
            // 
            picCanvas.BackColor = Color.Black;
            picCanvas.Location = new Point(0, 0);
            picCanvas.Name = "picCanvas";
            picCanvas.Size = new Size(280, 280);
            picCanvas.TabIndex = 0;
            picCanvas.TabStop = false;
            // 
            // numDigit
            // 
            numDigit.FormattingEnabled = true;
            numDigit.Items.AddRange(new object[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" });
            numDigit.Location = new Point(199, 15);
            numDigit.Name = "numDigit";
            numDigit.Size = new Size(81, 28);
            numDigit.TabIndex = 1;
            // 
            // btnClear
            // 
            btnClear.Anchor = AnchorStyles.Top;
            btnClear.Location = new Point(100, 15);
            btnClear.Name = "btnClear";
            btnClear.Size = new Size(94, 29);
            btnClear.TabIndex = 2;
            btnClear.Text = "Clear";
            btnClear.UseVisualStyleBackColor = true;
            btnClear.Click += btnClear_Click;
            // 
            // btnSave
            // 
            btnSave.Location = new Point(12, 242);
            btnSave.Name = "btnSave";
            btnSave.Size = new Size(94, 29);
            btnSave.TabIndex = 3;
            btnSave.Text = "Save";
            btnSave.UseVisualStyleBackColor = true;
            // 
            // btnPredict
            // 
            btnPredict.Location = new Point(178, 242);
            btnPredict.Name = "btnPredict";
            btnPredict.Size = new Size(94, 29);
            btnPredict.TabIndex = 4;
            btnPredict.Text = "Predict";
            btnPredict.UseVisualStyleBackColor = true;
            // 
            // lblStatus
            // 
            lblStatus.AutoSize = true;
            lblStatus.BackColor = Color.White;
            lblStatus.Font = new Font("Segoe UI", 6F, FontStyle.Regular, GraphicsUnit.Point, 0);
            lblStatus.Location = new Point(12, 216);
            lblStatus.Name = "lblStatus";
            lblStatus.Size = new Size(256, 12);
            lblStatus.TabIndex = 5;
            lblStatus.Text = "Draw a digit and select its label, then click Save or Predict.";
            // 
            // btnTrain
            // 
            btnTrain.Location = new Point(0, 14);
            btnTrain.Name = "btnTrain";
            btnTrain.Size = new Size(94, 29);
            btnTrain.TabIndex = 6;
            btnTrain.Text = "Train";
            btnTrain.UseVisualStyleBackColor = true;
            // 
            // CanvasForm
            // 
            AutoSize = true;
            AutoSizeMode = AutoSizeMode.GrowAndShrink;
            ClientSize = new Size(282, 283);
            Controls.Add(btnTrain);
            Controls.Add(lblStatus);
            Controls.Add(btnPredict);
            Controls.Add(btnSave);
            Controls.Add(btnClear);
            Controls.Add(numDigit);
            Controls.Add(picCanvas);
            Name = "CanvasForm";
            ((System.ComponentModel.ISupportInitialize)picCanvas).EndInit();
            ResumeLayout(false);
            PerformLayout();

        }

        /// <summary> 
        /// Converts the 280x280 canvas to a 784-element normalized double array (0.0 to 1.0) 
        /// </summary> 
        public double[] GetCanvasPixelArray()
        {
            // 1. Resize 280x280 down to 28x28
            Bitmap resized = new Bitmap(28, 28);
            using (Graphics g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(canvasBitmap, 0, 0, 28, 28);
            }

            // 2\. Extract normalized grayscale values
            double[] pixels = new double[784];
            int index = 0;

            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    Color pixelColor = resized.GetPixel(x, y);
                    // Average RGB values to get brightness, scale 0.0 (black) to 1.0 (white)
                    double brightness = (pixelColor.R + pixelColor.G + pixelColor.B) / 3.0 / 255.0;
                    pixels[index++] = Math.Round(brightness, 4);
                }
            }

            resized.Dispose();
            return pixels;
        }
    }
}
