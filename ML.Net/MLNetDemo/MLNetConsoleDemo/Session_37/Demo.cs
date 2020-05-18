using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLNetConsoleDemo.Session_37
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        private static readonly string demoPath = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../Session_37"));
        private static IDataView shuffledData;
        private static TrainTestData splitData;
        private static ITransformer model;

        public static void Execute()
        {
            LoadData();

            PreProcessData();

            CreateModel();

            EvaluateModel();

            Predict();
        }

        private static void LoadData()
        {
            var trainDatasetPath = Path.Combine(demoPath, "train-dataset");

            var images = LoadImagesFromFolder(trainDatasetPath);
            var dataview = context.Data.LoadFromEnumerable(images);
            shuffledData = context.Data.ShuffleRows(dataview);
        }

        private static void Predict()
        {
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var predictionDatasetPath = Path.Combine(demoPath, "prediction-dataset");
            var predictionImages = LoadImagesFromFolder(predictionDatasetPath, true);

            foreach (var image in predictionImages)
            {
                var result = predictionEngine.Predict(image);
                Console.WriteLine("Original Image: {0} | Prediction: {1} | Score: {2}",
                   image.FruitName, result.FruitName, string.Join(",", result.Score.Select(x => x.ToString("0.0000"))));
            }
        }

        private static void PreProcessData()
        {
            var trainDatasetPath = Path.Combine(demoPath, "train-dataset");

            // Pre-process Pipeline
            var preProcessingPipeline = context.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "Label", inputColumnName: nameof(InputModel.FruitName),
                    keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.LoadRawImageBytes(
                    outputColumnName: "Features", trainDatasetPath, nameof(InputModel.ImagePath)));

            // Pre-process data
            var PreProcessData = preProcessingPipeline.Fit(shuffledData)
                .Transform(shuffledData);
            splitData = context.Data.TrainTestSplit(PreProcessData);
        }

        private static void CreateModel()
        {
            if (File.Exists("ImageClassificationModel.zip"))
            {
                // If Model already exist then just load it
                using (FileStream fs = new FileStream("ImageClassificationModel.zip", FileMode.OpenOrCreate))
                    model = context.Model.Load(fs, out _);
            }
            else
            {
                // Create Pipeline
                var pipeline = context.MulticlassClassification.Trainers.ImageClassification()
                    .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                // Create Model
                model = pipeline.Fit(splitData.TrainSet);

                // And save it
                using (FileStream fs = new FileStream("ImageClassificationModel.zip", FileMode.OpenOrCreate))
                    context.Model.Save(model, splitData.TrainSet.Schema, fs);
            }
        }

        private static void EvaluateModel()
        {
            var testDataset = model.Transform(splitData.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(testDataset);
            Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy} | Micro Accuracy {metrics.MicroAccuracy}");
        }

        private static IEnumerable<InputModel> LoadImagesFromFolder(string datasetPath, bool loadBytes = false)
        {
            var allFiles = Directory.GetFiles(datasetPath, "*", searchOption: SearchOption.AllDirectories);
            var returnData = new List<InputModel>();

            foreach (string file in allFiles)
            {
                if (Path.GetExtension(file) != ".jpg")
                    continue;

                returnData.Add(new InputModel
                {
                    FruitName = Directory.GetParent(file).Name,
                    ImagePath = file,
                    ImageBytes = loadBytes ? File.ReadAllBytes(file) : null
                });
            }

            return returnData;
        }
    }
}
