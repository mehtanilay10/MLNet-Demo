using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace MLNetConsoleDemo.Session_44
{
    class Demo
    {
        private static readonly MLContext context = new MLContext();
        static IDataView dataview;

        static Demo()
        {
            var data = new List<InputModel>()
            {
                new InputModel{ Text = "First think another Disney movie, might good, it's kids movie. watch it, can't help enjoy it. ages love movie. first saw movie 10 8 years later still love it! Danny Glover superb could play part better. Christopher Lloyd hilarious perfect part. Tony Danza believable Mel Clark. can't help, enjoy movie! give 10/10!" },
                new InputModel{ Text = "Real classic. shipload sailors trying get towns daughters fathers go extremes deter sailors attempts. maidens cry aid results dispatch \"Rape Squad\". cult film waiting happen!" },
                new InputModel{ Text = "Great movie could Soylent Green. scenes people. people act 2022. think would neat see happen year 2022 beyond. Even still know secret great movie. go rent buy movie right NOW!!" },
                new InputModel{ Text = "Would rated film minus 10 sadly offered.<br /><br />Why didn't walk first five minutes movie cannot say. gone instinct left immediately!! Several people theater sadly didn't follow out.<br /><br />The story lacked criteria movie. plot. Awful acting! Even Robin Williams disappointing may never see another film in. single relationship story went beyond parlor talk. like tazer scene. bad didn't shock meat senselessness plot. Someone needs tazer writer director film!" },
            };

            dataview = context.Data.LoadFromEnumerable(data);
        }

        public static void Execute()
        {
            var preview = dataview.Preview();

            FeaturizeText();

            FeaturizeTextWithOptions();

            NormalizeText();
        }

        public static void FeaturizeText()
        {
            var pipeline = context.Transforms.Text.FeaturizeText("FeaturizedText", nameof(InputModel.Text));
            var p2 = pipeline.Fit(dataview).Transform(dataview).Preview();
            var featureColumn = (VBuffer<float>)p2.ColumnView[1].Values[0];
            var featureValues = featureColumn.GetValues();
        }

        private static void FeaturizeTextWithOptions()
        {
            var options = new TextFeaturizingEstimator.Options
            {
                CaseMode = TextNormalizingEstimator.CaseMode.Lower,
                KeepNumbers = false,
                KeepPunctuations = false,
                KeepDiacritics = false,
            };

            var pipeline = context.Transforms.Text.FeaturizeText("FeaturizedText", options: options,
                                                    inputColumnNames: new[] { nameof(InputModel.Text) });
            var p2 = pipeline.Fit(dataview).Transform(dataview).Preview();
            var featureColumn = (VBuffer<float>)p2.ColumnView[1].Values[0];
            var featureValues = featureColumn.GetValues();
        }

        private static void NormalizeText()
        {
            var pipeline = context.Transforms.Text.NormalizeText(
                outputColumnName: nameof(NormalizeResultModel.NormalizeText),
                inputColumnName: nameof(InputModel.Text),
                caseMode: TextNormalizingEstimator.CaseMode.Lower,
                keepDiacritics: false, keepPunctuations: false, keepNumbers: false);

            var model = pipeline.Fit(dataview);
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, NormalizeResultModel>(model);
            var res = predictionEngine.Predict(new InputModel { Text = "Some text with spaces and 10 digits, And also have some puncuations." });

            Console.WriteLine($"Original Text: {res.Text}");
            Console.WriteLine($"Normalized Text: {res.NormalizeText}");
        }
    }
}
