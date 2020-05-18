using Microsoft.ML;

namespace MLNetConsoleDemo.Session_70
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Training Data
            IDataView trainingData = context.Data.LoadFromTextFile<InputModel>(path: "Session_70/train-dataset.csv", hasHeader: true, separatorChar: ',');

            // Load ONNX Model
            var onnxEstimator = context.Transforms.ApplyOnnxModel("Session_70/SalaryPredictModel.onnx");

            // Add additional estimators
            var pipeline = onnxEstimator.Append(context.Transforms.CopyColumns(
                outputColumnName: nameof(ResultModel.SalaryCopied),
                inputColumnName: nameof(ResultModel.Salary)
            ));

            // Train Model
            var onnxModel = pipeline.Fit(trainingData);

            // Create Predaction Engine
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(onnxModel);

            // Predict data
            var experience = new InputModel { YearsOfExperience = 10 };

            var result = predictionEngine.Predict(experience);
            System.Console.WriteLine($"Approx Salary for {experience.YearsOfExperience} Years of experience will be: {result.Salary[0]}.");
        }
    }
}
