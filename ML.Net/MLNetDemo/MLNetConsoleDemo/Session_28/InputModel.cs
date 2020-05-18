using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_28
{
    class InputModel
    {
        [LoadColumn(0)]
        public bool Sentiment { get; set; }

        [LoadColumn(1)]
        public string SentimentText { get; set; }
    }
}
