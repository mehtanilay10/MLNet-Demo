using System.Linq;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_40
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Load data
            var dataview = context.Data.LoadFromTextFile<InputModel>("Session_40/train-dataset.csv", ',', true);
            var splitData = context.Data.TrainTestSplit(dataview);

            // Create Pipelines
            var dataPipeline = context.Transforms.Concatenate("Features", nameof(InputModel.Feature1), nameof(InputModel.Feature2),
                                                                            nameof(InputModel.Feature3), nameof(InputModel.Feature4))
                .Append(context.Transforms.Conversion.MapValueToKey(nameof(InputModel.Label)))
                .Append(context.Transforms.Conversion.Hash(nameof(InputModel.GroupId), nameof(InputModel.GroupId), 20));

            var trainer = context.Ranking.Trainers.FastTree();
            var pipeline = dataPipeline.Append(trainer);

            // Create Model
            var model = pipeline.Fit(splitData.TrainSet);

            // Evaluate Model
            IDataView predictions = model.Transform(splitData.TestSet);
            var eval = context.Ranking.Evaluate(predictions);
            System.Console.WriteLine("DCG: {0} | Normalized DCG: {1}",
                string.Join(", ", eval.DiscountedCumulativeGains.Select(x => x.ToString("N4"))),
                string.Join(", ", eval.NormalizedDiscountedCumulativeGains.Select(x => x.ToString("N4"))));

            // Predict value
            var predictEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);
            var result = predictEngine.Predict(new InputModel { Feature1 = 303F, Feature2 = 225F, Feature3 = 407F, Feature4 = 300F });
            System.Console.WriteLine($"Rank: {result.Label} | GroupId: {result.GroupId} | Score: {result.Score}");
        }
    }
}
