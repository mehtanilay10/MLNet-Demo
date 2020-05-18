using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNetConsoleDemo.Session_46
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

            WordBags();

            HashedWordBags();

            Ngrams();

            //HashedNgrams();
        }

        private static void WordBags()
        {
            var pipeline = context.Transforms.Text.ProduceWordBags(
                outputColumnName: nameof(WordBagResultModel.BagOfWordFeatures),
                inputColumnName: nameof(InputModel.Text));

            var model = pipeline.Fit(dataview);
            PrintFirstRowData(model, nameof(WordBagResultModel.BagOfWordFeatures));
        }

        private static void HashedWordBags()
        {
            var pipeline = context.Transforms.Text.ProduceHashedWordBags(
                outputColumnName: nameof(WordBagResultModel.BagOfWordFeatures),
                inputColumnName: nameof(InputModel.Text),
                numberOfBits: 5,
                ngramLength: 3,
                useAllLengths: false,
                maximumNumberOfInverts: 5);

            var model = pipeline.Fit(dataview);
            PrintFirstRowData(model, nameof(WordBagResultModel.BagOfWordFeatures));
        }

        private static void Ngrams()
        {
            var pipeline = context.Transforms.Text.TokenizeIntoWords(
                    outputColumnName: "Tokens",
                    inputColumnName: nameof(InputModel.Text))
                .Append(context.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "TokenizedKey",
                    inputColumnName: "Tokens"))
                .Append(context.Transforms.Text.ProduceNgrams(
                    outputColumnName: nameof(NgramResultModel.NgramFeatures),
                    inputColumnName: "TokenizedKey",
                    ngramLength: 3,
                    useAllLengths: false));

            var model = pipeline.Fit(dataview);
            PrintFirstRowData(model, nameof(NgramResultModel.NgramFeatures));
        }

        private static void HashedNgrams()
        {
            var pipeline = context.Transforms.Text.TokenizeIntoWords(
                    outputColumnName: "Tokens",
                    inputColumnName: nameof(InputModel.Text))
                .Append(context.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "TokenizedKey",
                    inputColumnName: "Tokens"))
                .Append(context.Transforms.Text.ProduceHashedNgrams(
                    outputColumnName: nameof(NgramResultModel.NgramFeatures),
                    inputColumnName: "TokenizedKey",
                    ngramLength: 3,
                    useOrderedHashing: true,
                    useAllLengths: false,
                    numberOfBits: 8));

            var model = pipeline.Fit(dataview);
            PrintFirstRowData(model, nameof(NgramResultModel.NgramFeatures));
        }

        private static void PrintFirstRowData(ITransformer model, string columnName)
        {
            // Transform Dataview
            var transformedDataView = model.Transform(dataview);

            // Obtain Slots
            VBuffer<ReadOnlyMemory<char>> slotNames = default;
            transformedDataView.Schema[columnName].GetSlotNames(ref slotNames);
            var slots = slotNames.GetValues();

            // Obtain Columns
            var collectionOfWords = transformedDataView.GetColumn<VBuffer<float>>(
                column: transformedDataView.Schema[columnName]);
            var textColumn = transformedDataView.GetColumn<string>(
                column: transformedDataView.Schema[nameof(InputModel.Text)]);

            // Print for 1st record
            var textRow = textColumn.FirstOrDefault();
            Console.WriteLine($"Text: {textRow} {Environment.NewLine}");

            var bagRow = collectionOfWords.FirstOrDefault();
            foreach (var item in bagRow.Items())
                Console.WriteLine(slots[item.Key]);
        }
    }
}
