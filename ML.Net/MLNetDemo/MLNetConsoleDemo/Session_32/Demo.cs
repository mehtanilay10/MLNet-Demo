using System;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_32
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Load data
            var trainingDataview = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_32/Amazon-reviews-train-dataset.csv", allowQuoting: true, hasHeader: true, separatorChar: ',');

            // Data Pipeline
            var dataPipeline = context.Transforms.Conversion.MapValueToKey("Label", nameof(InputModel.Ratings))
                .Append(context.Transforms.Text.FeaturizeText("Featurize_Summary", nameof(InputModel.Summary)))
                .Append(context.Transforms.Text.FeaturizeText("Featurize_ReviewText", nameof(InputModel.ReviewText)))
                .Append(context.Transforms.Concatenate("Features", "Featurize_Summary", "Featurize_ReviewText"));

            // Training Pipeline
            var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy(maximumNumberOfIterations: 50);
            var trainPipeline = dataPipeline.Append(trainer)
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Model
            var transformers = trainPipeline.Fit(trainingDataview);

            // Prediction Engine
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(transformers);

            PrintResult(predictionEngine.Predict(new InputModel
            {
                Summary = "Awesome Magazine",
                ReviewText = "I read this magazine and its pretty amazing.",
            }));

            PrintResult(predictionEngine.Predict(new InputModel
            {
                Summary = "Not A Good Deal, Disappointing content",
                ReviewText = "Not worth buying. Too many ads and tabloid style writing make this publication not worth your while.",
            }));
        }

        private static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.PredictedRating} | Score: {string.Join(",", result.Score)}");
        }
    }
}
