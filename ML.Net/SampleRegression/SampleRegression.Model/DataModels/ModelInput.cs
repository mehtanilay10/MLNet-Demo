//*****************************************************************************************
//*                                                                                       *
//* This is an auto-generated file by Microsoft ML.NET CLI (Command-Line Interface) tool. *
//*                                                                                       *
//*****************************************************************************************

using Microsoft.ML.Data;

namespace SampleRegression.Model.DataModels
{
    public class ModelInput
    {
        [ColumnName("YearsOfExperience"), LoadColumn(0)]
        public float YearsOfExperience { get; set; }


        [ColumnName("Salary"), LoadColumn(1)]
        public float Salary { get; set; }


    }
}
