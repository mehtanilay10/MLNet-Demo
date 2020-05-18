using Microsoft.ML.Data;

namespace MLNetASPDemo.Models
{
    public class ResultModel : InputModel
    {
        [ColumnName("Score")]
        public float Salary { get; set; }
    }
}
