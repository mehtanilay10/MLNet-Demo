using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace MLNetConsoleDemo.Session_41
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            // Load data
            var dataview = context.Data.LoadFromTextFile<InputModel>("Session_41/total-births-train-dataset.csv", ',', true);

            // Create Pipeline
            var pipeline = context.Forecasting.ForecastBySsa(
                outputColumnName: nameof(ResultModel.ForecastedBirths),
                inputColumnName: nameof(InputModel.Births),
                confidenceLevel: 0.95F,
                confidenceLowerBoundColumn: nameof(ResultModel.ConfidenceLowerBound),
                confidenceUpperBoundColumn: nameof(ResultModel.ConfidenceUpperBound),
                windowSize: 365,
                seriesLength: 365 * 3,
                trainSize: 365 * 3,
                horizon: 7
            );

            // Create Model
            var model = pipeline.Fit(dataview);

            // Forecast data
            var forecastEngine = model.CreateTimeSeriesEngine<InputModel, ResultModel>(context);
            var result = forecastEngine.Predict();

            System.Console.WriteLine($"Next 7 day Forecast: [{string.Join(", ", result.ForecastedBirths.Select(x => x.ToString("0")))}]");

            // Save Model
            forecastEngine.CheckPoint(context, "TimeSeriesModel.zip");
        }
    }
}
