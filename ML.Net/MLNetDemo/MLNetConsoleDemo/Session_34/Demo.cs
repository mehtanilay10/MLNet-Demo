using System.Linq;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_34
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Load data
            var dataview = context.Data.LoadFromTextFile<InputModel>("Session_34/train-dataset.csv", ',', true);
            var splitDataview = context.Data.TrainTestSplit(dataview, testFraction: 0.01);

            // Create Pipeline
            var pipeline = context.Transforms.Conversion.MapValueToKey("Label", nameof(InputModel.Class))
                .Append(context.MulticlassClassification.Trainers.LightGbm(numberOfLeaves: 100))
                .Append(context.Transforms.Conversion.MapKeyToValue(nameof(ResultModel.PredictedClass), nameof(ResultModel.PredictedLabel)));

            // Train the model.
            var model = pipeline.Fit(splitDataview.TrainSet);

            // Verify with testset
            var testData = model.Transform(splitDataview.TestSet);
            var predictions = context.Data.CreateEnumerable<ResultModel>(testData, reuseRowObject: false).ToList();
            foreach (var prediction in predictions)
            {
                System.Console.WriteLine($"Existing Class: {prediction.Class} | Predicted Class: {prediction.PredictedClass}");
            }

            // Evaluate the trained model is the test set.
            var metrics = context.MulticlassClassification.Evaluate(testData);

            // Check if metrics are resonable.
            System.Console.WriteLine($"Macro accuracy: {metrics.MacroAccuracy}, Micro accuracy: {metrics.MicroAccuracy}.");
        }
    }
}
