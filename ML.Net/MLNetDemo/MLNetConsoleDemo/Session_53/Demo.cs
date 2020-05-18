using Microsoft.ML;

namespace MLNetConsoleDemo.Session_53
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_53/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            OneHotHashEncoding();

            OneHotHashEncodingMultipleColumn();
        }

        private static void OneHotHashEncoding()
        {
            var pipeline = context.Transforms.Categorical.OneHotHashEncoding("HashEncoded_City", nameof(InputModel.City),
                outputKind: Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Binary,
                numberOfBits: 8, useOrderedHashing: false);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void OneHotHashEncodingMultipleColumn()
        {
            var columns = new InputOutputColumnPair[]
            {
                new InputOutputColumnPair("Encoded_City", nameof(InputModel.City)),
                new InputOutputColumnPair("Encoded_IsProUser", nameof(InputModel.IsProUser)),
            };

            var pipeline = context.Transforms.Categorical.OneHotHashEncoding(columns: columns,
                outputKind: Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Binary,
                numberOfBits: 31, useOrderedHashing: true);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
