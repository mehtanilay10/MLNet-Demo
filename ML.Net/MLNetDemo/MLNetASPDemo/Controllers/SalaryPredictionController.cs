using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using MLNetASPDemo.Models;

namespace MLNetASPDemo.Controllers
{
    [ApiController]
    public class SalaryPredictionController : ControllerBase
    {
        private readonly PredictionEnginePool<InputModel, ResultModel> predictionEnginePool;

        public SalaryPredictionController(PredictionEnginePool<InputModel, ResultModel> predictionEnginePool)
        {
            this.predictionEnginePool = predictionEnginePool;
        }

        [Route("Predict")]
        public ActionResult<string> Predict(int experience)
        {
            InputModel input = new InputModel { YearsOfExperience = experience };

            ResultModel result = predictionEnginePool.Predict(modelName: "SalaryPredictModel", example: input);

            var msg = $"Approx Salary for {result.YearsOfExperience} Years of experience will be: {result.Salary}.";

            return Ok(msg);
        }
    }
}
