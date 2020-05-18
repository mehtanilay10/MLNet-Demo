using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_40
{
    class InputModel
    {
        [LoadColumn(0)]
        public float Label { get; set; }

        [LoadColumn(1)]
        public float GroupId { get; set; }

        [LoadColumn(2)]
        public float Feature1 { get; set; }

        [LoadColumn(3)]
        public float Feature2 { get; set; }

        [LoadColumn(4)]
        public float Feature3 { get; set; }

        [LoadColumn(5)]
        public float Feature4 { get; set; }
    }
}
