using System.Collections.Generic;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_50
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_50/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            ValueToArray();

            ValueLookup();
        }

        private static void ValueToArray()
        {
            var dictionary = new Dictionary<string, int[]>
            {
                {"San Francisco", new[] { 5, 8, 7, 9} },
                {"Chicago", new [] { 8, 7, 3, 1} },
                {"Washington", new [] { 10, 5, 7, 6} },
                {"Williamsburg", new [] { 3, 8, 1, 12} },
                {"Seattle", new[] { 8, 6, 9, 18} },
                {"Las Vegas", new [] { 15, 12, 7, 4} },
                {"Boston", new [] { 13, 4, 6, 8} }
            };

            var pipeline = context.Transforms.Conversion.MapValue(
                    outputColumnName: "Features",
                    inputColumnName: nameof(InputModel.City),
                    keyValuePairs: dictionary);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void ValueLookup()
        {
            var lookup = new List<ProUserLookup>
            {
                new ProUserLookup{ IsProUser = true, DisplayText ="Premium User"},
                new ProUserLookup{ IsProUser = false, DisplayText = "Normal User"}
            };

            var lookupDataview = context.Data.LoadFromEnumerable(lookup);

            var pipeline = context.Transforms.Conversion.ConvertType("Bool_IsProUser", nameof(InputModel.IsProUser), Microsoft.ML.Data.DataKind.Boolean)
                .Append(context.Transforms.Conversion.MapValue(
                    outputColumnName: "UserTypeText",
                    inputColumnName: "Bool_IsProUser",
                    lookupMap: lookupDataview,
                    keyColumn: lookupDataview.Schema[nameof(ProUserLookup.IsProUser)],
                    valueColumn: lookupDataview.Schema[nameof(ProUserLookup.DisplayText)]));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
