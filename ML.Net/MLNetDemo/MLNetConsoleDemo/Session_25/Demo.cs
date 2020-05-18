using Microsoft.ML;

namespace MLNetConsoleDemo.Session_25
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Training Data
            IDataView trainingData = context.Data.LoadFromTextFile<InputModel>(path: "Session_25/train-dataset.csv", hasHeader: true, separatorChar: ',');

            // Prepare data 
            var estimator = context.Transforms.Concatenate("Features", new[] { "YearsOfExperience" });

            // Create pipeline
            var pipeline = estimator.Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));

            // Train Model
            var model = pipeline.Fit(trainingData);

            // Save Model
            if (!System.IO.File.Exists("Session_25/SalaryPredictModel.zip"))
                context.Model.Save(model, trainingData.Schema, "Session_25/SalaryPredictModel.zip");

            // Load saved model
            var savedModel = context.Model.Load("Session_25/SalaryPredictModel.zip", out DataViewSchema schema);

            //ITransformer savedModel;
            //using (var fileStream = new FileStream("Session_25/SalaryPredictModel.zip", FileMode.OpenOrCreate))
            //    savedModel = context.Model.Load(fileStream, out DataViewSchema schema);

            //var savedModel = context.Model.LoadWithDataLoader("Session_25/SalaryPredictModel.zip", out IDataLoader<IMultiStreamSource> loader);

            // Create Predaction Engine
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(savedModel);

            // Predict data
            var experience = new InputModel { YearsOfExperience = 10 };

            var result = predictionEngine.Predict(experience);
            System.Console.WriteLine($"Approx Salary for {experience.YearsOfExperience} Years of experience will be: {result.Salary}.");
        }
    }
}
