using System;
using System.Collections.Generic;
using Microsoft.ML.Transforms.TimeSeries;

namespace MLNetConsoleDemo.Session_60
{
    static class PrintHelper
    {
        public static void PrintChangePointResult(TimeSeriesPredictionEngine<InputModel, ResultModel> engine, IEnumerable<int> indexes)
        {
            Console.WriteLine("Index\tAlert\tScore\tP-Value\t\tMartingale value");
            foreach (var value in indexes)
            {
                var res = engine.Predict(new InputModel(value));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}\t\t{4:0.00}",
                    value, res.Prediction[0], res.Prediction[1], res.Prediction[2], res.Prediction[3]);
            }
        }
    }
}
