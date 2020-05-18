using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace MLNetConsoleDemo.Session_30
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Load data
            var dataset = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_30/titanic-passenger-train-dataset.csv", hasHeader: true, separatorChar: ',', allowQuoting: true);
            var trainAndTestDataset = context.Data.TrainTestSplit(dataset);

            // Generate Data pipeline
            var dataPipeline = context.Transforms.Conversion.ConvertType(
                                    outputColumnName: "Numeric_Age",
                                    inputColumnName: nameof(InputModel.Age),
                                    outputKind: Microsoft.ML.Data.DataKind.Single)
                .Append(context.Transforms.ReplaceMissingValues(
                                    outputColumnName: "Numeric_Age_without_MissingValue",
                                    inputColumnName: "Numeric_Age",
                                    replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_Gender", nameof(InputModel.Gender)))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_Embarked", nameof(InputModel.Embarked)))
                .Append(context.Transforms.Concatenate("Features", "Numeric_Age_without_MissingValue", "Encoded_Gender",
                                    "Encoded_Embarked", nameof(InputModel.ParCh), nameof(InputModel.SibSp), nameof(InputModel.Fare)));

            // Append Algorithm
            //var trainer = context.BinaryClassification.Trainers.FastTree(new FastTreeBinaryTrainer.Options
            //{
            //    LabelColumnName = nameof(InputModel.Survived),
            //    NumberOfTrees = 50,
            //});

            var trainer = context.BinaryClassification.Trainers.FastForest(new FastForestBinaryTrainer.Options
            {
                LabelColumnName = nameof(InputModel.Survived),
                NumberOfTrees = 50,
            });

            // Create Model
            var trainingPipeline = dataPipeline.Append(trainer);
            var model = trainingPipeline.Fit(trainAndTestDataset.TrainSet);

            var preview = model.Transform(dataset).Preview();

            // Evaluate
            //var metrics = context.BinaryClassification.Evaluate(model.Transform(trainAndTestDataset.TestSet),
            //    labelColumnName: nameof(InputModel.Survived));
            var metrics = context.BinaryClassification.EvaluateNonCalibrated(model.Transform(trainAndTestDataset.TestSet),
                labelColumnName: nameof(InputModel.Survived));
        }
    }
}
