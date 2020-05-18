using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_33
{
    class ResultModel : InputModel
    {
        [ColumnName("PredictedLabel")]
        public uint Prediction { get; set; }
    }
}
