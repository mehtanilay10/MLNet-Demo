using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_24
{
    class InputModel
    {
        [LoadColumn(0)]
        public float YearsOfExperience { get; set; }

        [LoadColumn(1)]
        public float Salary { get; set; }
    }
}
