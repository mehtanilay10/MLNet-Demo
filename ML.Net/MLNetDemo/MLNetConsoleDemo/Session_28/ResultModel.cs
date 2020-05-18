using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_28
{
    class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public bool IsPositiveReview { get; set; }

        public float Score { get; set; }
    }
}
