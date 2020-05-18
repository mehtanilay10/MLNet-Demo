using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_33
{
    class InputModel
    {
        [LoadColumn(0)]
        public uint Label { get; set; }

        [VectorType(8)]
        [LoadColumn(new[] { 1, 2, 3, 4, 5, 6, 7, 8 })]
        public float[] Features { get; set; }
    }
}
