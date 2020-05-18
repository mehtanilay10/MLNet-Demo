using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_58
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview;

        static Demo()
        {
            var data = new List<InputModel>
            {
                new InputModel
                {
                    Features1 = new float[] { float.PositiveInfinity, 18F, 0, 12 },
                    Features2 = new float[] { 4, float.NaN, 6 }
                },
                new InputModel
                {
                    Features1 = new float[] { float.NaN, 18, 0, 8 },
                    Features2 = new float[] { 4, 5, 6 }
                },
                new InputModel
                {
                    Features1 = new float[] {  0, 8, 7, 10  },
                    Features2 = new float[] { 8, 3, 1 }
                },
                new InputModel
                {
                    Features1 = new float[] { 20, 10, 0, 17F },
                    Features2 = new float[] { 8, 6, 4 }
                },
                new InputModel
                {
                    Features1 = new float[] { 15, 0, 10, 14F },
                    Features2 = new float[] { 4, 0, float.NaN }
                }
            };
            dataview = context.Data.LoadFromEnumerable(data);
        }

        public static void Execute()
        {
            ReplaceMissingValues();

            ReplaceMissingValuesMultipleColumns();

            IndicateMissingValues();

            IndicateMissingValuesMultipleColumns();
        }

        private static void ReplaceMissingValues()
        {
            var pipeline = context.Transforms.ReplaceMissingValues(outputColumnName: nameof(ReplaceResultModel.WithoutMissingValues1),
                    inputColumnName: nameof(InputModel.Features1),
                    replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.DefaultValue)
                .Append(context.Transforms.ReplaceMissingValues(outputColumnName: nameof(ReplaceResultModel.WithoutMissingValues2),
                    inputColumnName: nameof(InputModel.Features2),
                    replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var convertedList = context.Data.CreateEnumerable<ReplaceResultModel>(convertedDataview, false).ToList();
            PrintFeatures(convertedList);
        }

        private static void ReplaceMissingValuesMultipleColumns()
        {
            var columns = new InputOutputColumnPair[]
            {
                new InputOutputColumnPair(nameof(ReplaceResultModel.WithoutMissingValues1), nameof(InputModel.Features1)),
                new InputOutputColumnPair(nameof(ReplaceResultModel.WithoutMissingValues2), nameof(InputModel.Features2)),
            };

            var pipeline = context.Transforms.ReplaceMissingValues(columns: columns,
                replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Maximum);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var convertedList = context.Data.CreateEnumerable<ReplaceResultModel>(convertedDataview, false).ToList();
            PrintFeatures(convertedList);
        }

        private static void IndicateMissingValues()
        {
            var pipeline = context.Transforms.IndicateMissingValues(outputColumnName: nameof(IndicateResultModel.IsMissingFeature1),
                    inputColumnName: nameof(InputModel.Features1))
                .Append(context.Transforms.IndicateMissingValues(outputColumnName: nameof(IndicateResultModel.IsMissingFeature2),
                    inputColumnName: nameof(InputModel.Features2)));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var convertedList = context.Data.CreateEnumerable<IndicateResultModel>(convertedDataview, false).ToList();
            PrintFeatures(convertedList);
        }

        private static void IndicateMissingValuesMultipleColumns()
        {
            var columns = new InputOutputColumnPair[]
            {
                new InputOutputColumnPair(nameof(IndicateResultModel.IsMissingFeature1), nameof(InputModel.Features1)),
                new InputOutputColumnPair(nameof(IndicateResultModel.IsMissingFeature2), nameof(InputModel.Features2)),
            };

            var pipeline = context.Transforms.IndicateMissingValues(columns: columns);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var convertedList = context.Data.CreateEnumerable<IndicateResultModel>(convertedDataview, false).ToList();
            PrintFeatures(convertedList);
        }

        private static void PrintFeatures(List<ReplaceResultModel> result)
        {
            Console.WriteLine("All Features");
            var f1 = result.Select(x => string.Join(",\t", x.Features1)).ToList();
            var f2 = result.Select(x => string.Join(",\t", x.Features2)).ToList();

            for (int i = 0; i < result.Count; i++)
                Console.WriteLine($"{f1[i]}\t\t{f2[i]}");

            Console.WriteLine("After Replacing Missing Values");
            f1 = result.Select(x => string.Join(",\t", x.WithoutMissingValues1)).ToList();
            f2 = result.Select(x => string.Join(",\t", x.WithoutMissingValues2)).ToList();

            for (int i = 0; i < result.Count; i++)
                Console.WriteLine($"{f1[i]}\t\t{f2[i]}");
        }

        private static void PrintFeatures(List<IndicateResultModel> result)
        {
            Console.WriteLine("All Features");
            var f1 = result.Select(x => string.Join(",\t", x.Features1)).ToList();
            var f2 = result.Select(x => string.Join(",\t", x.Features2)).ToList();

            for (int i = 0; i < result.Count; i++)
                Console.WriteLine($"{f1[i]}\t\t{f2[i]}");

            Console.WriteLine("Is Missing Values");
            f1 = result.Select(x => string.Join(",\t", x.IsMissingFeature1)).ToList();
            f2 = result.Select(x => string.Join(",\t", x.IsMissingFeature2)).ToList();

            for (int i = 0; i < result.Count; i++)
                Console.WriteLine($"{f1[i]}\t\t{f2[i]}");
        }
    }
}
