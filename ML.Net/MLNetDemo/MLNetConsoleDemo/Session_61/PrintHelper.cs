using System;
using System.Collections.Generic;
using Microsoft.ML.Transforms.TimeSeries;

namespace MLNetConsoleDemo.Session_61
{
    static class PrintHelper
    {
        public static void PrintAnomalyResult(TimeSeriesPredictionEngine<InputModel, ResultModel> engine, IEnumerable<int> indexes, int input)
        {
            Console.WriteLine("Input\tIndex\tAlert\tScore\tMartingale value");
            foreach (var value in indexes)
            {
                var res = engine.Predict(new InputModel(input));
                Console.WriteLine("{0}\t{1}\t{2}\t{3:0.00}\t{4:0.00}",
                                input, value, res.Prediction[0], res.Prediction[1], res.Prediction[2]);
            }
        }

        public static void PrintSpikeResult(TimeSeriesPredictionEngine<InputModel, ResultModel> engine, IEnumerable<int> indexes)
        {
            Console.WriteLine("Index\tAlert\tScore\tP-value");
            foreach (var value in indexes)
            {
                var res = engine.Predict(new InputModel(value));
                Console.WriteLine("{0}\t{1}\t{2:0.00}\t{3:0.00}",
                                value, res.Prediction[0], res.Prediction[1], res.Prediction[2]);
            }
        }
    }
}
