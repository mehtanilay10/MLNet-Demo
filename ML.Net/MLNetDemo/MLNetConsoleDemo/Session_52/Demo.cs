using Microsoft.ML;

namespace MLNetConsoleDemo.Session_52
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_52/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            OneHotEncoding();

            OneHotEncodingMultipleColumn();
        }

        private static void OneHotEncoding()
        {
            var pipeline = context.Transforms.Categorical.OneHotEncoding("Encoded_City", nameof(InputModel.City),
                outputKind: Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Binary,
                keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void OneHotEncodingMultipleColumn()
        {
            var columns = new InputOutputColumnPair[]
            {
                new InputOutputColumnPair("Encoded_City", nameof(InputModel.City)),
                new InputOutputColumnPair("Encoded_IsProUser", nameof(InputModel.IsProUser)),
            };

            var pipeline = context.Transforms.Categorical.OneHotEncoding(columns: columns,
                outputKind: Microsoft.ML.Transforms.OneHotEncodingEstimator.OutputKind.Binary,
                keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByOccurrence);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
