using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_39
{
    class ResultModel : InputModel
    {
        [ColumnName("PredictedLabel")]
        public bool HasAnomaly { get; set; }

        public float Score { get; set; }
    }
}
