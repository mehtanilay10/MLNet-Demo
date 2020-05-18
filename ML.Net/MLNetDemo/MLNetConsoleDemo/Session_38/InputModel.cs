using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_38
{
    class InputModel
    {
        [LoadColumn(0)]
        public string ImageName { get; set; }
    }
}
