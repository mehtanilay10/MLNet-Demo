using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MLNetConsoleDemo.Session_39
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Load data
            var trainDataview = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_39/student-marks-train-dataset.csv", separatorChar: ',', hasHeader: true);

            // Create Pipeline
            var pipeline = context.AnomalyDetection.Trainers.RandomizedPca(new RandomizedPcaTrainer.Options
            {
                FeatureColumnName = nameof(InputModel.Marks),
                Rank = 1,
            });

            // Create Model
            var model = pipeline.Fit(trainDataview);

            // Run Model on Trainset
            var transformedDataview = model.Transform(trainDataview);
            var results = context.Data.CreateEnumerable<ResultModel>(transformedDataview, reuseRowObject: false).ToList();

            foreach (var student in results)
            {
                System.Console.WriteLine("Has Anomaly: {0} | Score: {1} | Subject Name: {2} | Marks: [{3}]",
                    student.HasAnomaly, student.Score, student.SubjectName, string.Join(", ", student.Marks.Select(x => x.ToString("0"))));
            }
        }
    }
}
