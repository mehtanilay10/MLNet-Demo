using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_27
{
    class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public bool WillDelayBy15Minutes { get; set; }

        public float Score { get; set; }
    }
}
