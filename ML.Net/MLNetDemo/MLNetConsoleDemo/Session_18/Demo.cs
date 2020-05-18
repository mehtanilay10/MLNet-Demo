using Microsoft.ML;

namespace MLNetConsoleDemo.Session_18
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            IDataView dataView = context.Data.LoadFromBinary("Session_18/train-dataset.bin");

            var preview = dataView.Preview();
        }
    }
}
