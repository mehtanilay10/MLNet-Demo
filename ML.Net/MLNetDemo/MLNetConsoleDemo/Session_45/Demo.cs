using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace MLNetConsoleDemo.Session_45
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

            dataview = context.Data.LoadFromEnumerable<InputModel>(data);
        }

        public static void Execute()
        {
            var preview = dataview.Preview();

            Tokenize();

            TokenizeIntoCharacters();

            RemovedDefaultStopWords();

            RemoveCustomStopWords();
        }

        private static void Tokenize()
        {
            var pipeline = context.Transforms.Text.TokenizeIntoWords(
                    outputColumnName: nameof(TokenizeResultModel.Tokens),
                    inputColumnName: nameof(InputModel.Text));

            var model = pipeline.Fit(dataview);
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, TokenizeResultModel>(model);
            var res = predictionEngine.Predict(new InputModel { Text = "Some text with spaces and 10 digits, And also have some puncuations." });

            Console.WriteLine($"Original Text: {res.Text}");
            Console.WriteLine(string.Join(Environment.NewLine, res.Tokens));
        }

        private static void TokenizeIntoCharacters()
        {
            var pipeline = context.Transforms.Text.TokenizeIntoCharactersAsKeys(
                    outputColumnName: "CharKeys",
                    inputColumnName: nameof(InputModel.Text))
                .Append(context.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: nameof(TokenizeResultModel.Tokens),
                    inputColumnName: "CharKeys"
                ));

            var model = pipeline.Fit(dataview);
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, TokenizeResultModel>(model);
            var res = predictionEngine.Predict(new InputModel { Text = "Some text with spaces and 10 digits, And also have some puncuations." });

            Console.WriteLine($"Original Text: {res.Text}");
            Console.WriteLine(string.Join(Environment.NewLine, res.Tokens));
        }

        private static void RemovedDefaultStopWords()
        {
            var pipeline = context.Transforms.Text.TokenizeIntoWords(
                    outputColumnName: nameof(StopWordResultModel.Tokens),
                    inputColumnName: nameof(InputModel.Text))
                .Append(context.Transforms.Text.RemoveDefaultStopWords(
                     outputColumnName: nameof(StopWordResultModel.AfterRemovingDefaultStopWords),
                     inputColumnName: nameof(StopWordResultModel.Tokens)));

            var model = pipeline.Fit(dataview);
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, StopWordResultModel>(model);
            var res = predictionEngine.Predict(new InputModel { Text = "Some text with spaces and 10 digits, And also have some puncuations." });

            Console.WriteLine($"Original Text: {res.Text}");
            Console.WriteLine($"After Removing Stopwords: {string.Join(" ", res.AfterRemovingDefaultStopWords)}");
        }

        private static void RemoveCustomStopWords()
        {
            string[] stopwords = new string[] { "with", "and", "have" };
            var pipeline = context.Transforms.Text.TokenizeIntoWords(
                    outputColumnName: nameof(StopWordResultModel.Tokens),
                    inputColumnName: nameof(InputModel.Text))
                .Append(context.Transforms.Text.RemoveStopWords(
                    outputColumnName: nameof(StopWordResultModel.AfterRemovingDefaultStopWords),
                    inputColumnName: nameof(StopWordResultModel.Tokens),
                    stopwords: stopwords));

            var model = pipeline.Fit(dataview);
            var predictionEngine = context.Model.CreatePredictionEngine<InputModel, StopWordResultModel>(model);
            var res = predictionEngine.Predict(new InputModel { Text = "Some text with spaces and 10 digits, And also have some puncuations." });

            Console.WriteLine($"Original Text: {res.Text}");
            Console.WriteLine($"After Removing Stopwords: {string.Join(" ", res.AfterRemovingDefaultStopWords)}");
        }
    }
}
