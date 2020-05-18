namespace MLNetConsoleDemo.Session_45
{
    class StopWordResultModel : InputModel
    {
        public string[] Tokens { get; set; }
        public string[] AfterRemovingDefaultStopWords { get; set; }
    }
}
