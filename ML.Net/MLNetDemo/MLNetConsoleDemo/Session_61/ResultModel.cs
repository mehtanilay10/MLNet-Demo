using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_61
{
    class ResultModel
    {
        [VectorType(4)]
        public double[] Prediction { get; set; }
    }
}
