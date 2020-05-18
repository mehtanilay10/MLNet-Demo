using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace MLNetConsoleDemo.Session_48
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_48/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            MapToKey();

            MapToValue();

            MapMultipleColumn();
        }

        private static void MapToKey()
        {
            var pipeline = context.Transforms.Conversion.MapValueToKey("Key_Name", nameof(InputModel.Name),
                    keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void MapToValue()
        {
            var pipeline = context.Transforms.Conversion.MapValueToKey("Key_Name", nameof(InputModel.Name),
                    keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.Conversion.MapKeyToValue("Value_Name", "Key_Name"));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void MapMultipleColumn()
        {
            var pipeline = context.Transforms.Conversion.MapValueToKey(
                    new InputOutputColumnPair[]
                    {
                        new InputOutputColumnPair("Key_Name", nameof(InputModel.Name)),
                        new InputOutputColumnPair("Key_Amount", nameof(InputModel.AmountPaid))
                    })
                .Append(context.Transforms.Conversion.MapKeyToValue(
                    new InputOutputColumnPair[]
                    {
                        new InputOutputColumnPair("Value_Name", "Key_Name"),
                        new InputOutputColumnPair("Value_Amount", "Key_Amount")
                    }));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
