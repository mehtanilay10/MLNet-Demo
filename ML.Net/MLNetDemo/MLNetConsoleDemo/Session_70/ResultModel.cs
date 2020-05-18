using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_70
{
    class ResultModel
    {
        [ColumnName("Score")]
        public float[] Salary { get; set; }

        public float[] SalaryCopied { get; set; }
    }
}
