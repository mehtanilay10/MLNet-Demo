using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_32
{
    class InputModel
    {
        [LoadColumn(1)]
        public string Summary { get; set; }

        [LoadColumn(2)]
        public string ReviewText { get; set; }

        [LoadColumn(3)]
        public int Ratings { get; set; }
    }
}
