using System.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_19
{
    class Demo
    {
        public static void Execute()
        {
            MLContext context = new MLContext();

            var databaseLoader = context.Data.CreateDatabaseLoader<InputModel>();

            string connectionString = "Data Source=MNILAY-ENVY\\SQLEXPRESS;Initial Catalog=MLDemoData;Integrated Security=True";
            string commandText = "SELECT sd.YearsOfExperience, sd.Salary FROM dbo.SalaryData sd";

            var databaseSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, commandText);
            IDataView dataView = databaseLoader.Load(databaseSource);

            var preview = dataView.Preview();
        }
    }
}
