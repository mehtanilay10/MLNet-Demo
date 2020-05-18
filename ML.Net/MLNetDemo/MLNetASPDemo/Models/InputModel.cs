using Microsoft.ML.Data;

namespace MLNetASPDemo.Models
{
    public class InputModel
    {
        [LoadColumn(0)]
        public float YearsOfExperience { get; set; }

        [LoadColumn(1)]
        public float Salary { get; set; }
    }
}
