using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_29
{
    class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedRecommendation { get; set; }

        public float Score { get; set; }
    }
}
