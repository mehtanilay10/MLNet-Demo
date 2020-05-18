using System;
using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_41
{
    class InputModel
    {
        [LoadColumn(0)]
        public DateTime Date { get; set; }

        [LoadColumn(1)]
        public float Births { get; set; }
    }
}
