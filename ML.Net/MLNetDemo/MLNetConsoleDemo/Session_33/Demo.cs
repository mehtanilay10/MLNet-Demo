using System.Linq;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_33
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext(1);

            // Load data
            var dataview = context.Data.LoadFromTextFile<InputModel>("Session_33/train-dataset.csv", ',', true);
            var trainAndTestDataview = context.Data.TrainTestSplit(dataview, testFraction: 0.02);

            // Create Pipeline
            var binaryTrainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();

            var pipeline = context.Transforms.Conversion.MapValueToKey(nameof(InputModel.Label))
                .Append(context.MulticlassClassification.Trainers.OneVersusAll(binaryTrainer));

            // Create Model
            var model = pipeline.Fit(trainAndTestDataview.TrainSet);

            // Verify with testset
            var testData = model.Transform(trainAndTestDataview.TestSet);
            var predictions = context.Data.CreateEnumerable<ResultModel>(testData, reuseRowObject: false).ToList();
            foreach (var prediction in predictions)
            {
                System.Console.WriteLine($"Original value: {prediction.Label} | Predicted value: {prediction.Prediction}");
            }
        }
    }
}
