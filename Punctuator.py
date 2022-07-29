import  requests


def Punctuate(unsegmented_text):
    data = {
        'text' : unsegmented_text
    }
    URL = "http://bark.phon.ioc.ee/punctuator"
    # sending get request and saving the response as response object
    r = requests.post(url=URL, data=data)

    # extracting data in json format
    data = r.text
    return data










