using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_32
{
    class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public int PredictedRating { get; set; }

        public float[] Score { get; set; }
    }
}
