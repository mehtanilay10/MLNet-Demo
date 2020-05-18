using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_27
{
    class InputModel
    {
        [LoadColumn(0)]
        public string Origin { get; set; }

        [LoadColumn(1)]
        public string Destination { get; set; }

        [LoadColumn(2)]
        public float DepartureTime { get; set; }

        [LoadColumn(3)]
        public float ExpectedArrivalTime { get; set; }

        [LoadColumn(4)]
        public float OriginalArrivalTime { get; set; }

        [LoadColumn(5)]
        public int DelayMinutes { get; set; }

        [LoadColumn(6)]
        public bool IsDelayBy15Minutes { get; set; }
    }
}
