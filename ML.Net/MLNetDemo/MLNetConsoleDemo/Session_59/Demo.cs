using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_59
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview;

        static Demo()
        {
            var data = new List<InputModel>
            {
                new InputModel(10, 18, 6, 12),
                new InputModel(15, 18, 2,  8),
                new InputModel( 8,  8, 7, 10),
                new InputModel(20, 10, 5, 17),
                new InputModel(15, 7, 10, 14)
            };
            dataview = context.Data.LoadFromEnumerable(data);
        }

        public static void Execute()
        {
            Expression();

            NormalizeMinMax();
        }

        private static void Expression()
        {
            var pipeline = context.Transforms.Expression(outputColumnName: "Expr1Value",
                    expression: "(x, y) => x + y",
                    inputColumnNames: new[] { nameof(InputModel.Feature1), nameof(InputModel.Feature2) })
                .Append(context.Transforms.Expression(outputColumnName: "Expr2Value",
                    expression: "(x, y, z) => min(x, y) + log(z)",
                    inputColumnNames: new[] { nameof(InputModel.Feature1), nameof(InputModel.Feature2), nameof(InputModel.Feature3) }));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void NormalizeMinMax()
        {
            var pipeline = context.Transforms.Concatenate("Features", nameof(InputModel.Feature1), nameof(InputModel.Feature2),
                                                                        nameof(InputModel.Feature3), nameof(InputModel.Feature4))
                .Append(context.Transforms.NormalizeMinMax(outputColumnName: "NormalizedFeature",
                    fixZero: false, inputColumnName: "Features"));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);

            var normalizedValues = convertedDataview.GetColumn<float[]>("NormalizedFeature");
            foreach (var row in normalizedValues)
                Console.WriteLine(string.Join(",\t", row.Select(x => x.ToString("F4"))));
        }
    }
}
