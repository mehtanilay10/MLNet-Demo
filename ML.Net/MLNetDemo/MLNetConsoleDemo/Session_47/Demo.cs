using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace MLNetConsoleDemo.Session_47
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_47/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            ConvertType();

            ConvertTypeMultipleColumn();

            Hash();

            HashMultipleColumn();
        }

        private static void ConvertType()
        {
            var pipeline = context.Transforms.Conversion.ConvertType(
                outputColumnName: "Converted_IsPro",
                inputColumnName: nameof(InputModel.IsProUser),
                outputKind: Microsoft.ML.Data.DataKind.Boolean);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void ConvertTypeMultipleColumn()
        {
            var columns = new InputOutputColumnPair[] {
                new InputOutputColumnPair("Converted_IsPro", nameof(InputModel.IsProUser)),
                new InputOutputColumnPair("Converted_Age", nameof(InputModel.Age)),
                new InputOutputColumnPair("Converted_Date", nameof(InputModel.JoiningDate)),
                new InputOutputColumnPair("Converted_Amount", nameof(InputModel.AmountPaid)),
            };

            var pipeline = context.Transforms.Conversion.ConvertType(columns, Microsoft.ML.Data.DataKind.Single);
            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void Hash()
        {
            var pipeline = context.Transforms.Conversion.Hash("Hashed_Name", nameof(InputModel.Name), numberOfBits: 10);
            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void HashMultipleColumn()
        {
            var columns = new HashingEstimator.ColumnOptions[] {
                new HashingEstimator.ColumnOptions("Hashed_Name", nameof(InputModel.Name), 20),
                new HashingEstimator.ColumnOptions("Hashed_Age", nameof(InputModel.Age), 10)
            };

            var pipeline = context.Transforms.Conversion.Hash(columns);
            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
