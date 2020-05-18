using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_14
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // 1] Import data 
            List<InputModel> data = new List<InputModel>
            {
                new InputModel { YearsOfExperience = 1, Salary= 39000 },
                new InputModel { YearsOfExperience = 1.3F, Salary= 46200 },
                new InputModel { YearsOfExperience = 1.5F, Salary= 37700 },
                new InputModel { YearsOfExperience = 2, Salary= 43500 },
                new InputModel { YearsOfExperience = 2.2F, Salary= 40000 },
                new InputModel { YearsOfExperience = 2.9F, Salary= 56000 },
                new InputModel { YearsOfExperience = 3, Salary= 60000 },
                new InputModel { YearsOfExperience = 3.2F, Salary= 54000 },
                new InputModel { YearsOfExperience = 3.3F, Salary= 64000 },
                new InputModel { YearsOfExperience = 3.7F, Salary= 57000 },
                new InputModel { YearsOfExperience = 3.9F, Salary= 63000 },
                new InputModel { YearsOfExperience = 4, Salary= 55000 },
                new InputModel { YearsOfExperience = 4, Salary= 58000 },
                new InputModel { YearsOfExperience = 4.1F, Salary= 57000 },
                new InputModel { YearsOfExperience = 4.5F, Salary= 61000 },
                new InputModel { YearsOfExperience = 4.9F, Salary= 68000 },
                new InputModel { YearsOfExperience = 5.3F, Salary= 83000 },
                new InputModel { YearsOfExperience = 5.9F, Salary= 82000 },
                new InputModel { YearsOfExperience = 6, Salary= 94000 },
                new InputModel { YearsOfExperience = 6.8F, Salary= 91000 },
                new InputModel { YearsOfExperience = 7.1F, Salary= 98000 },
                new InputModel { YearsOfExperience = 7.9F, Salary= 101000 },
                new InputModel { YearsOfExperience = 8.2F, Salary= 114000 },
                new InputModel { YearsOfExperience = 8.9F, Salary= 109000 },
            };

            // 2] Training Data
            IDataView trainingData = context.Data.LoadFromEnumerable(data);

            // 3] Prepare data 
            var estimator = context.Transforms.Concatenate("Features", new[] { "YearsOfExperience" });

            // 4] Create pipeline
            //var pipeline = estimator.Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));
            //var pipeline = estimator.Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 10));
            //var pipeline = estimator.Append(context.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100, l1Regularization: 2, l2Regularization: 2));
            //var pipeline = estimator.Append(context.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Salary"));
            //var pipeline = estimator.Append(context.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Salary", optimizationTolerance: 100));
            //var pipeline = estimator.Append(context.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Salary"));
            var pipeline = estimator.Append(context.Regression.Trainers.OnlineGradientDescent(labelColumnName: "Salary", numberOfIterations: 100));

            // 5] Train Model
            var model = pipeline.Fit(trainingData);

            // 6] Create Test data
            List<InputModel> testData = new List<InputModel>
            {
                new InputModel { YearsOfExperience = 1.3F, Salary= 46200 },
                new InputModel { YearsOfExperience = 2.9F, Salary= 56000 },
                new InputModel { YearsOfExperience = 3.2F, Salary= 54000 },
                new InputModel { YearsOfExperience = 3.9F, Salary= 63000 },
                new InputModel { YearsOfExperience = 4.1F, Salary= 57000 },
                new InputModel { YearsOfExperience = 7.1F, Salary= 98000 },
            };

            // 7] Load Test data
            var testDataView = context.Data.LoadFromEnumerable(testData);

            // 8] Evaluate Model
            var metrics = context.Regression.Evaluate(model.Transform(testDataView), labelColumnName: "Salary");

            Console.WriteLine($"R^2: {metrics.RSquared:0.00}");
        }
    }
}
