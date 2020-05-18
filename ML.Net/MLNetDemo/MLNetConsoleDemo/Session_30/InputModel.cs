using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_30
{
    class InputModel
    {
        [LoadColumn(1)]
        public bool Survived { get; set; }

        [LoadColumn(4)]
        public string Gender { get; set; }

        [LoadColumn(5)]
        public string Age { get; set; }

        [LoadColumn(6)]
        public float SibSp { get; set; }

        [LoadColumn(7)]
        public float ParCh { get; set; }

        [LoadColumn(9)]
        public float Fare { get; set; }

        [LoadColumn(11)]
        public string Embarked { get; set; }
    }
}
