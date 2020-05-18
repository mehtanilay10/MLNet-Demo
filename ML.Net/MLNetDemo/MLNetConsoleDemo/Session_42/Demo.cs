using Microsoft.ML;

namespace MLNetConsoleDemo.Session_42
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Load data
            var dataview = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_42/youtube-treanding-video-train-dataset.csv", separatorChar: ',', hasHeader: true);
            var splitData = context.Data.TrainTestSplit(dataview);

            // Create pipeline
            var pipeline = context.Transforms.Concatenate("Features", nameof(InputModel.Views),
                    nameof(InputModel.Likes), nameof(InputModel.Dislikes), nameof(InputModel.Comments))
                .Append(context.Clustering.Trainers.KMeans(options: new Microsoft.ML.Trainers.KMeansTrainer.Options
                {
                    NumberOfClusters = 100,
                    FeatureColumnName = "Features",
                }));

            // Create Model
            var model = pipeline.Fit(splitData.TrainSet);

            // Evaluate
            var testData = model.Transform(splitData.TestSet);
            var result = context.Clustering.Evaluate(testData);
            System.Console.WriteLine($"Avg. Distance: {result.AverageDistance}");

            // Predict
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);
            PrintResult(predictionEngine.Predict(new InputModel
            {
                Comments = 100,
                Dislikes = 10000,
                Likes = 100,
                Views = 12000
            }));

            PrintResult(predictionEngine.Predict(new InputModel
            {
                Comments = 15000,
                Dislikes = 100,
                Likes = 200000,
                Views = 400000
            }));
        }

        public static void PrintResult(ResultModel result)
        {
            System.Console.WriteLine($"Prediction: {result.PredictedCluster} | Score: [{string.Join(",", result.Distances)}]");
        }
    }
}
