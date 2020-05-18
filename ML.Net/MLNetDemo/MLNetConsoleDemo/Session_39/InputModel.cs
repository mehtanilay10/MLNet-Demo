using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_39
{
    class InputModel
    {
        [LoadColumn(0)]
        public string SubjectName { get; set; }

        [VectorType(10)]
        [LoadColumn(new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 })]
        public float[] Marks { get; set; }
    }
}
