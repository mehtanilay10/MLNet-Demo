using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_57
{
    class InputModel
    {
        [VectorType(4)]
        public float[] Features1 { get; set; }

        [VectorType(3)]
        public float[] Features2 { get; set; }
    }
}
