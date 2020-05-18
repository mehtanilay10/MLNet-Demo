using System.IO;
using System.Linq;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_20
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            var textLoader = context.Data.CreateTextLoader<InputModel>(separatorChar: ',');
            IDataView dataView = textLoader.Load(
                "Session_20/training-dataset/01.csv",
                "Session_20/training-dataset/02.csv");

            var list = context.Data.CreateEnumerable<InputModel>(dataView, false).ToList();

            using (FileStream stream = new FileStream("Session_20/combined-dataset.tsv", FileMode.OpenOrCreate))
            {
                context.Data.SaveAsText(dataView, stream);
            }

            using (FileStream stream = new FileStream("Session_20/combined-dataset.csv", FileMode.OpenOrCreate))
            {
                context.Data.SaveAsText(dataView, stream, separatorChar: ',', headerRow: false, schema: false);
            }

            using (FileStream stream = new FileStream("Session_20/combined-dataset.bin", FileMode.OpenOrCreate))
            {
                context.Data.SaveAsBinary(dataView, stream);
            }
        }
    }
}
