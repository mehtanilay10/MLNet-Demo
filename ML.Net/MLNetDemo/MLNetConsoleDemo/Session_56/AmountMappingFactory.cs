using System;
using Microsoft.ML.Transforms;

namespace MLNetConsoleDemo.Session_56
{
    [CustomMappingFactoryAttribute("IsAmountMoreThan3K")]
    class AmountMappingFactory : CustomMappingFactory<InputModel, MappingResult>
    {
        Action<InputModel, MappingResult> mappingAction = (input, output) => output.IsAmountMoreThan3K = (input.AmountPaid > 3000);

        public override Action<InputModel, MappingResult> GetMapping()
        {
            return mappingAction;
        }
    }
}
