using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace MLNetConsoleDemo.Session_38
{
    class Demo
    {
        private readonly static MLContext context = new MLContext();
        private static readonly string demoPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../Session_38"));
        private static ITransformer model;
        private static EstimatorChain<ImagePixelExtractingTransformer> preProcessPipeline;

        public static void Execute()
        {
            PreProcessData();

            LoadModel();

            PredictLabels();
        }

        private static void PreProcessData()
        {
            var imagesFolder = Path.Combine(demoPath, "Images");

            preProcessPipeline = context.Transforms.LoadImages(outputColumnName: "image",
                    imageFolder: imagesFolder, inputColumnName: nameof(InputModel.ImageName))
                .Append(context.Transforms.ResizeImages(outputColumnName: "resizedImage",
                    imageWidth: Constants.ImageWidth, imageHeight: Constants.ImageHeight, inputColumnName: "image"))
                .Append(context.Transforms.ExtractPixels(outputColumnName: Constants.TFInputColumnName,
                    interleavePixelColors: true, offsetImage: Constants.MeanValue, inputColumnName: "resizedImage"));
        }

        private static void LoadModel()
        {
            var tfPipeline = context.Model.LoadTensorFlowModel(Path.Combine(demoPath, "tensorflow-model.pb"))
                .ScoreTensorFlowModel(outputColumnName: Constants.TFOutputColumnName, inputColumnName: Constants.TFInputColumnName, true);

            var estimator = preProcessPipeline.Append(tfPipeline);
            var emptyDataView = context.Data.LoadFromEnumerable(new List<InputModel>());
            model = estimator.Fit(emptyDataView);
        }

        private static void PredictLabels()
        {
            var allLabels = File.ReadAllLines(Path.Combine(demoPath, "tensorflow-labels.txt"));
            var testDataview = context.Data.LoadFromTextFile<InputModel>("Session_38/test-dataset.csv");
            var testData = context.Data.CreateEnumerable<InputModel>(testDataview, false).ToList();

            var predictEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            foreach (var input in testData)
            {
                var result = predictEngine.Predict(input);
                var bestPrediction = result.PredictedLabels.Max();
                var predictedLabel = allLabels[result.PredictedLabels.AsSpan().IndexOf(bestPrediction)];
                Console.WriteLine($"Image: {input.ImageName} | Predicted: {predictedLabel} | Probablity: {bestPrediction}");
            }
        }
    }
}
