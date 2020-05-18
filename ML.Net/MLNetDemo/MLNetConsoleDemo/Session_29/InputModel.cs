using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_29
{
    class InputModel
    {
        [LoadColumn(2)]
        public string Summary { get; set; }

        [LoadColumn(3)]
        public string ReviewText { get; set; }

        [LoadColumn(4)]
        public bool Recommend { get; set; }
    }
}
