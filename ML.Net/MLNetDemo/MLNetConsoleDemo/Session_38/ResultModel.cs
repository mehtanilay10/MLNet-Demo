using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_38
{
    class ResultModel
    {
        [ColumnName(Constants.TFOutputColumnName)]
        public float[] PredictedLabels;
    }
}
