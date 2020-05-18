using System.IO;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_69
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Training Data
            IDataView trainingData = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_69/train-dataset.csv", hasHeader: true, separatorChar: ',');

            // Prepare data 
            var estimator = context.Transforms.Concatenate("Features", new[] { nameof(InputModel.YearsOfExperience) });

            // Create pipeline
            var pipeline = estimator.Append(context.Regression.Trainers.Sdca(
                                                labelColumnName: nameof(InputModel.Salary), maximumNumberOfIterations: 100));

            // Train Model
            var model = pipeline.Fit(trainingData);

            // Save Model
            using (var fileStream = File.Create("Session_69/SalaryPredictModel.onnx"))
                context.Model.ConvertToOnnx(model, trainingData, fileStream);
        }
    }
}
