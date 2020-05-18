using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_42
{
    class InputModel
    {
        [LoadColumn(2)]
        public float Views { get; set; }

        [LoadColumn(3)]
        public float Likes { get; set; }

        [LoadColumn(4)]
        public float Dislikes { get; set; }

        [LoadColumn(5)]
        public float Comments { get; set; }
    }
}
