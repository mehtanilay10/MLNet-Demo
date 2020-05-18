using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using static Microsoft.ML.DataOperationsCatalog;

namespace MLNetConsoleDemo.Session_43
{
    class Demo
    {
        private static MLContext context = new MLContext();
        private static IDataView dataview;
        private static TrainTestData splitData;
        private static ITransformer model;
        private static EstimatorChain<ValueToKeyMappingTransformer> estimator;

        public static void Execute()
        {
            LoadData();

            PreProcessData();

            CreateModel();

            EvaluateModel();

            PredictValue();
        }

        private static void LoadData()
        {
            dataview = context.Data.LoadFromTextFile<InputModel>("Session_43/book-ratings-train-dataset.csv", ',', true);
        }

        private static void PreProcessData()
        {
            // Pre-process Pipeline
            estimator = context.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "Encoded_UserID", inputColumnName: nameof(InputModel.UserId))
                .Append(context.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "Encoded_Book", inputColumnName: nameof(InputModel.ISBN)));

            // Pre-process data
            var preProcessData = estimator.Fit(dataview)
                .Transform(dataview);

            splitData = context.Data.TrainTestSplit(preProcessData, 0.05);
        }

        private static void CreateModel()
        {
            var options = new MatrixFactorizationTrainer.Options
            {
                LabelColumnName = nameof(InputModel.Rating),
                MatrixColumnIndexColumnName = "Encoded_UserID",
                MatrixRowIndexColumnName = "Encoded_Book",
                NumberOfIterations = 100,
                ApproximationRank = 100
            };

            var trainer = context.Recommendation().Trainers.MatrixFactorization(options);
            var pipeline = estimator.Append(trainer);
            model = pipeline.Fit(splitData.TrainSet);
        }

        private static void EvaluateModel()
        {
            var predictions = model.Transform(splitData.TestSet);
            var metrics = context.Recommendation().Evaluate(predictions, labelColumnName: nameof(InputModel.Rating));
            Console.WriteLine("R^2 {0} | LossFunction: {1} | MeanAbsoluteError: {2} | MeanSquaredError: {3}",
                metrics.RSquared, metrics.LossFunction, metrics.MeanAbsoluteError, metrics.MeanSquaredError);
        }

        private static void PredictValue()
        {
            var predictEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            PrintResult(predictEngine.Predict(new InputModel { UserId = 12, ISBN = "1879384493" }));
            PrintResult(predictEngine.Predict(new InputModel { UserId = 12, ISBN = "425176428" }));
        }

        private static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"UserId: {result.UserId} | Book: {result.ISBN} | Score: {result.Score} : Is Recommended: {result.Score > 7}");
        }
    }
}
