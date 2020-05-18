using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_24
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Training Data
            IDataView trainingData = context.Data.LoadFromTextFile<InputModel>(path: "Session_24/train-dataset.csv", hasHeader: true, separatorChar: ',');

            // Prepare data 
            var estimator = context.Transforms.Concatenate("Features", new[] { "YearsOfExperience" });

            // Create pipeline
            var pipeline = estimator.Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));

            // Train Model
            var model = pipeline.Fit(trainingData);

            // Save Model
            //context.Model.Save(model, trainingData.Schema, "Session_24/SalaryPredictModel.zip");

            //using (var fileStream = new FileStream("Session_24/SalaryPredictModel.zip", FileMode.Create))
            //    context.Model.Save(model, trainingData.Schema, fileStream);

            var file = new MultiFileSource("Session_24/train-dataset.csv");
            var dataLoader = context.Data.CreateTextLoader<InputModel>(hasHeader: true, separatorChar: ',', dataSample: file);
            context.Model.Save(model, dataLoader, "Session_24/SalaryPredictModel.zip");
        }
    }
}
