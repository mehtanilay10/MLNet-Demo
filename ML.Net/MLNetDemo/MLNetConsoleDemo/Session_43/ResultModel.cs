using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_43
{
    class ResultModel : InputModel
    {
        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
