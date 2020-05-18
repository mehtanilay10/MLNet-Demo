using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_37
{
    class ResultModel
    {
        [ColumnName("PredictedLabel")]
        public string FruitName { get; set; }

        public float[] Score { get; set; }
    }
}
