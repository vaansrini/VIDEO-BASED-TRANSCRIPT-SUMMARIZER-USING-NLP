import  requests

from deepsegment import DeepSegment

def Punctuate(unsegmented_text):
    # The default language is 'en'
    segmenter = DeepSegment('en')
    data = segmenter.segment(unsegmented_text , check_point = 'finetuned')
    punctuated = ""
    for i in range(len(data)):
        punctuated+=data[i]
        punctuated+='.'
    return punctuated

























































































