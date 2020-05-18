using Microsoft.ML;

namespace MLNetConsoleDemo.Session_16
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            //var columnsToLoad = new TextLoader.Column[]
            //{
            //    new TextLoader.Column("YearsOfExperience", DataKind.Single, 0),
            //    new TextLoader.Column("Salary", DataKind.Single, 1),
            //};

            //IDataView dataView = context.Data.LoadFromTextFile(
            //    path: "Session_16/train-dataset.csv", hasHeader: true, separatorChar: ',', columns: columnsToLoad);

            //IDataView dataView = context.Data.LoadFromTextFile<InputModel>(
            //    path: "Session_16/train-dataset.csv", hasHeader: true, separatorChar: ',');

            IDataView dataView = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_16/train-dataset.tsv", hasHeader: true);

            var preview = dataView.Preview();
        }
    }
}
