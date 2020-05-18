using Microsoft.ML;

namespace MLNetConsoleDemo.Session_49
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_49/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            MapToVector();

            MapToBinaryVector();
        }

        private static void MapToVector()
        {
            var pipeline = context.Transforms.Conversion.MapValueToKey("Key_Name", nameof(InputModel.Name))
                .Append(context.Transforms.Conversion.MapKeyToVector("Vector_Name", "Key_Name"));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void MapToBinaryVector()
        {
            var pipeline = context.Transforms.Conversion.MapValueToKey("Key_Name", nameof(InputModel.Name))
                .Append(context.Transforms.Conversion.MapKeyToBinaryVector("Binary_Name", "Key_Name"));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
