using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace MLNetConsoleDemo.Session_60
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();

        public static void Execute()
        {
            DetectChangePoint();
        }

        private static void DetectChangePoint()
        {
            var data = new List<InputModel>();
            for (int i = 0; i < 4; i++)
            {
                data.Add(new InputModel(10));
                data.Add(new InputModel(12));
                data.Add(new InputModel(14));
                data.Add(new InputModel(16));
            }

            var dataview = context.Data.LoadFromEnumerable(data);

            var pipeline = context.Transforms.DetectChangePointBySsa(outputColumnName: nameof(ResultModel.Prediction),
                    inputColumnName: nameof(InputModel.Value),
                    confidence: 95,
                    changeHistoryLength: 8,
                    trainingWindowSize: (4 * 5),
                    seasonalityWindowSize: 4);

            var model = pipeline.Fit(dataview);
            var engine = model.CreateTimeSeriesEngine<InputModel, ResultModel>(context);
            PrintHelper.PrintChangePointResult(engine, new List<int> { 1, 2, 3, 4, 5 });
            PrintHelper.PrintChangePointResult(engine, new List<int> { 100, 200, 300, 400, 500 });
        }
    }
}
