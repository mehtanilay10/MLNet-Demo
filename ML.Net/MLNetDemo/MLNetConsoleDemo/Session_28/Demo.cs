using System;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_28
{
    class Demo
    {
        public static void Execute()
        {
            // Create new MLContext
            MLContext context = new MLContext();

            // Load data
            var trainingDataView = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_28/IMDB-reviews-train-dataset.tsv", hasHeader: true);

            // Prepare data & create pipeline
            var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(InputModel.SentimentText))
                .Append(context.BinaryClassification.Trainers.AveragedPerceptron(
                    labelColumnName: nameof(InputModel.Sentiment), numberOfIterations: 100));

            // Train Model
            var model = pipeline.Fit(trainingDataView);

            // Create predator 
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var input = new InputModel { SentimentText = "I liked this movie." };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { SentimentText = "Movie was just ok." };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { SentimentText = "It's a really good film. Outragously entertaining, loved it." };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { SentimentText = "It's worst movie I have ever seen. Boring, badly written." };
            PrintResult(predictionEngine.Predict(input));
        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.IsPositiveReview} | Score: {result.Score}");
        }
    }
}
