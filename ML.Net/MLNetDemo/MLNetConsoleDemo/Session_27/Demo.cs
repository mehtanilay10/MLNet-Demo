using System;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_27
{
    class Demo
    {
        public static void Execute()
        {
            // Create new MLContext
            MLContext context = new MLContext();

            // Load data
            var dataView = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_26/Flight-Delay-train-dataset.csv", hasHeader: true, separatorChar: ',');

            // Prepare data & create pipeline
            var pipeline = context.Transforms.SelectColumns(nameof(InputModel.Origin), nameof(InputModel.Destination),
                                                            nameof(InputModel.DepartureTime), nameof(InputModel.ExpectedArrivalTime),
                                                            nameof(InputModel.IsDelayBy15Minutes))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_ORIGIN", nameof(InputModel.Origin)))
                .Append(context.Transforms.Categorical.OneHotEncoding("Encoded_DESTINATION", nameof(InputModel.Destination)))
                .Append(context.Transforms.DropColumns(nameof(InputModel.Origin), nameof(InputModel.Destination)))
                .Append(context.Transforms.Concatenate("Features", "Encoded_ORIGIN", "Encoded_DESTINATION",
                                                            nameof(InputModel.DepartureTime), nameof(InputModel.ExpectedArrivalTime)))
                .Append(context.Transforms.Conversion.ConvertType("Label", nameof(InputModel.IsDelayBy15Minutes), Microsoft.ML.Data.DataKind.Boolean))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            // Train Model
            var model = pipeline.Fit(dataView);
            var preview = model.Transform(dataView).Preview();

            // Create predator 
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, ResultModel>(model);

            var input = new InputModel { Origin = "JFK", Destination = "ATL", DepartureTime = 1930, ExpectedArrivalTime = 2225 };
            PrintResult(predictionEngine.Predict(input));

            input = new InputModel { Origin = "MSP", Destination = "SEA", DepartureTime = 1745, ExpectedArrivalTime = 1930 };
            PrintResult(predictionEngine.Predict(input));
        }

        static void PrintResult(ResultModel result)
        {
            Console.WriteLine($"Prediction: {result.WillDelayBy15Minutes} | Score: {result.Score}");
        }
    }
}
