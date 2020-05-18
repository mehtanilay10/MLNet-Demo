using Microsoft.ML;

namespace MLNetConsoleDemo.Session_54
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_54/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            CopyColumns();

            Concatenate();
        }

        private static void CopyColumns()
        {
            var pipeline = context.Transforms.CopyColumns("FullName", nameof(InputModel.Name))
                .Append(context.Transforms.CopyColumns("BillingDate", nameof(InputModel.JoiningDate)));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void Concatenate()
        {
            var pipeline = context.Transforms.Concatenate("ConcatenatedData", nameof(InputModel.Name), nameof(InputModel.City));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
