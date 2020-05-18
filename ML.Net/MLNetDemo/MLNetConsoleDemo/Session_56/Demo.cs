using System;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_56
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview = context.Data.LoadFromTextFile<InputModel>("Session_56/userdetails-train-dataset.csv", ',', true);

        public static void Execute()
        {
            CustomMapping();

            CustomMappingWithContract();
        }

        private static void CustomMapping()
        {
            Action<InputModel, MappingResult> mappingAction = (input, output) => output.IsAmountMoreThan3K = (input.AmountPaid > 3000);
            var pipeline = context.Transforms.CustomMapping(mappingAction, contractName: null);

            var model = pipeline.Fit(dataview);
            var convertedDataview = model.Transform(dataview);
            var preview = convertedDataview.Preview();
        }

        private static void CustomMappingWithContract()
        {
            var pipeline = context.Transforms.CustomMapping(new AmountMappingFactory().GetMapping(), contractName: "IsAmountMoreThan3K");
            var model = pipeline.Fit(dataview);

            context.ComponentCatalog.RegisterAssembly(typeof(AmountMappingFactory).Assembly);
            context.Model.Save(model, dataview.Schema, "Session_56/Model.zip");
            var savedModel = context.Model.Load("Session_56/Model.zip", out var inputSchema);

            var convertedDataview = savedModel.Transform(dataview);
            var preview = convertedDataview.Preview();
        }
    }
}
