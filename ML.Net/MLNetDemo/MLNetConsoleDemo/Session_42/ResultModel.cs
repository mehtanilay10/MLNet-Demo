using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_42
{
    class ResultModel : InputModel
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedCluster { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }
}
