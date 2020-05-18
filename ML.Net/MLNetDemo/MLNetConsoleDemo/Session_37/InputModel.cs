using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_37
{
    class InputModel
    {
        public string ImagePath { get; set; }
        public string FruitName { get; set; }

        [ColumnName("Features")]
        public byte[] ImageBytes { get; set; }
    }
}
