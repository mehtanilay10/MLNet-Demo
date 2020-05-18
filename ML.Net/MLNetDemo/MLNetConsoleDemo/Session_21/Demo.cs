using Microsoft.ML;

namespace MLNetConsoleDemo.Session_21
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            IDataView dataView = context.Data.LoadFromTextFile<InputModel>(
                path: "Session_21/train-dataset.csv", hasHeader: true, separatorChar: ',');
            var preview = dataView.Preview();

            var suffledData = context.Data.ShuffleRows(dataView);
            preview = suffledData.Preview();

            var skipedData = context.Data.SkipRows(dataView, 8);
            preview = skipedData.Preview();

            var takeData = context.Data.TakeRows(dataView, 8);
            preview = takeData.Preview();

            var filterByValue = context.Data.FilterRowsByColumn(dataView, nameof(InputModel.YearsOfExperience), lowerBound: 3, upperBound: 6);
            preview = filterByValue.Preview();

            var filterByMissingValue = context.Data.FilterRowsByMissingValues(dataView, nameof(InputModel.Salary));
            preview = filterByMissingValue.Preview();
        }
    }
}
