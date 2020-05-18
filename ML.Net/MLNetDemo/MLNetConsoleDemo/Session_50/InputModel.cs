using System;
using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_50
{
    class InputModel
    {
        [LoadColumn(1)]
        public string Name { get; set; }

        [LoadColumn(2)]
        public int Age { get; set; }

        [LoadColumn(3)]
        public int IsProUser { get; set; }

        [LoadColumn(4)]
        public float AmountPaid { get; set; }

        [LoadColumn(5)]
        public DateTime JoiningDate { get; set; }

        [LoadColumn(6)]
        public string City { get; set; }
    }
}
