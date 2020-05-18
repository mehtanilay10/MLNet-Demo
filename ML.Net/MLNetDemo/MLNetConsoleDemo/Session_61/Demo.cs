using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace MLNetConsoleDemo.Session_61
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();

        public static void Execute()
        {
            DetectAnomaly();

            DetectSpike();
        }

        private static void DetectAnomaly()
        {
            var data = new List<InputModel>();
            for (int i = 0; i < 30; i++)
                data.Add(new InputModel(10));
            data.Insert(20, new InputModel(100));            // Add Anomaly

            var dataview = context.Data.LoadFromEnumerable(data);

            var pipeline = context.Transforms.DetectAnomalyBySrCnn(outputColumnName: nameof(ResultModel.Prediction),
                    inputColumnName: nameof(InputModel.Value),
                    threshold: 0.35,
                    windowSize: 16,
                    judgementWindowSize: 8);

            var model = pipeline.Fit(dataview);
            var engine = model.CreateTimeSeriesEngine<InputModel, ResultModel>(context);
            PrintHelper.PrintAnomalyResult(engine, Enumerable.Range(1, 30), 10);
            PrintHelper.PrintAnomalyResult(engine, Enumerable.Range(1, 1), 100);
        }

        private static void DetectSpike()
        {
            var data = new List<InputModel>();
            for (int i = 0; i < 4; i++)
            {
                data.Add(new InputModel(1));
                data.Add(new InputModel(2));
                data.Add(new InputModel(3));
                data.Add(new InputModel(4));
                data.Add(new InputModel(5));
            }

            var dataview = context.Data.LoadFromEnumerable(data);

            var pipeline = context.Transforms.DetectSpikeBySsa(outputColumnName: nameof(ResultModel.Prediction),
                    inputColumnName: nameof(InputModel.Value),
                    confidence: 95,
                    pvalueHistoryLength: 5,
                    trainingWindowSize: (4 * 5),
                    seasonalityWindowSize: 5);

            var model = pipeline.Fit(dataview);
            var engine = model.CreateTimeSeriesEngine<InputModel, ResultModel>(context);
            PrintHelper.PrintSpikeResult(engine, new List<int> { 1, 2, 3, 4, 1, 2, 3, 400, 420, 430, 450, 1 });
        }
    }
}
