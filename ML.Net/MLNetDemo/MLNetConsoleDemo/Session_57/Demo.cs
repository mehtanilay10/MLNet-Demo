using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_57
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
                    Features1 = new float[] { 10, 18F, 0, 12 },
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
            SelectBasedOnCount();

            SelectBasedOnCountMultipleColumns();
        }

        private static void SelectBasedOnCount()
        {
            var pipeline = context.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(
                    outputColumnName: nameof(ResultModel.SelectedFeatures1),
                    inputColumnName: nameof(InputModel.Features1),
                    count: 3)
                .Append(context.Transforms.CopyColumns(
                    outputColumnName: nameof(ResultModel.SelectedFeatures2),
                    inputColumnName: nameof(InputModel.Features2)));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            PrintFeatures(convertedDataview);
        }

        private static void SelectBasedOnCountMultipleColumns()
        {
            var columns = new InputOutputColumnPair[]
            {
                new InputOutputColumnPair(nameof(ResultModel.SelectedFeatures1), nameof(InputModel.Features1)),
                new InputOutputColumnPair(nameof(ResultModel.SelectedFeatures2), nameof(InputModel.Features2)),
            };

            var pipeline = context.Transforms.FeatureSelection.SelectFeaturesBasedOnCount(
                    columns: columns,
                    count: 4);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            PrintFeatures(convertedDataview);
        }

        private static void PrintFeatures(IDataView convertedDataview)
        {
            var convertedList = context.Data.CreateEnumerable<ResultModel>(convertedDataview, false).ToList();

            Console.WriteLine("All Features");
            var f1 = convertedList.Select(x => string.Join(",\t", x.Features1)).ToList();
            var f2 = convertedList.Select(x => string.Join(",\t", x.Features2)).ToList();

            for (int i = 0; i < convertedList.Count; i++)
                Console.WriteLine($"{f1[i]}\t\t{f2[i]}");

            Console.WriteLine("Selected Features");
            f1 = convertedList.Select(x => string.Join(",\t", x.SelectedFeatures1)).ToList();
            f2 = convertedList.Select(x => string.Join(",\t", x.SelectedFeatures2)).ToList();

            for (int i = 0; i < convertedList.Count; i++)
                Console.WriteLine($"{f1[i]}\t\t{f2[i]}");
        }
    }
}
