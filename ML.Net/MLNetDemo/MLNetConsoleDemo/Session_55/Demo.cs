using Microsoft.ML;

namespace MLNetConsoleDemo.Session_55
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_55/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            SelectColumns();

            DropColumns();

            DropColumnsAfterProcessing();
        }

        private static void SelectColumns()
        {
            var pipeline = context.Transforms.SelectColumns(nameof(InputModel.Name), nameof(InputModel.Age),
                nameof(InputModel.City), nameof(InputModel.IsProUser));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void DropColumns()
        {
            var pipeline = context.Transforms.DropColumns(nameof(InputModel.AmountPaid), nameof(InputModel.JoiningDate));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void DropColumnsAfterProcessing()
        {
            var pipeline = context.Transforms.Text.TokenizeIntoWords("Tokenized_Name", nameof(InputModel.Name))
                .Append(context.Transforms.DropColumns(nameof(InputModel.Name)));

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
