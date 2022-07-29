import sys
import validators
from online_transcript_extraction import ExtractYoutubeTrancript
from offline_video_to_transcript import OfflineVideoTranscript
from Punctuator import Punctuate
from TF_IDF import TF_IDF
from Lex_Rank import lex_rank
from Text_Rank import TextRank
from RogueMetrics import RogueMetrics
from os import path
# Press the green button in the gutter to run the scrip

if __name__ == '__main__':
    url = sys.argv[1]
    if(validators.url(url)):
        example_url = url
        _id = example_url.split("=")[1].split("&")[0]
        print(_id)
        Transcript = ExtractYoutubeTrancript(_id)
    elif(path.exists(url)):
        Transcript = OfflineVideoTranscript(url)
    else:
        print("Given Input is neither a video path nor a valid url")
        exit(0)
    Punctuated_transcript = Punctuate(Transcript)
    percent = input("Percentage of Summary : ")
    print("Actual Content :  \n")
    print(Punctuated_transcript)
    tf_idf_summary = TF_IDF(Punctuated_transcript,percent)
    lex_rank_summary = lex_rank(Punctuated_transcript,percent)
    BetterSummary = RogueMetrics(Punctuated_transcript,tf_idf_summary,lex_rank_summary , percent)
    for sentence in BetterSummary:
        print(sentence)







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
