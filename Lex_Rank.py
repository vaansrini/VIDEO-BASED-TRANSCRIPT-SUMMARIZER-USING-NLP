import numpy as np
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from scipy import spatial
from gensim.summarization.summarizer import summarize
import networkx as nx


def tokenize_docs(corpus, tokenizer):
    """Take a corpus of documents, tokenize and vectorize them
    INPUT: corpus (list of strings): list of documents to be processed
    tokenizer (tokenizer class): NLTK tokenizer
    OUTPUT: tfidf_matrix (sparse numpy matrix): tfidf-vectorized representation
    of a document from the corpus
    """
    # stemmed_docs will be a list of lists
    stemmed_docs = []
    # produce a list of lists of tokenized words, representing words in the
    # corpus
    for document in corpus:
        stemmed_docs.append([])
        for word in document.split():
            stemmed_docs[-1].append(tokenizer.stem(word))

    # take a random document in the corpus to process
    feed_string = ' '.join(stemmed_docs[0])
    sentences = punkt.tokenize(feed_string)

    tfidf_matrix = tfidf_vect.fit_transform(sentences)
    return tfidf_matrix, sentences

def calculate_similarity_mat(tfidf_matrix,
                                 similarity_threshold=0.05):
        """Produce a matrix representation of a graph of document similarities
        INPUT: tfidf_test (numpy matrix): a tf_idf representation of the corpus to
        work with
        similarity_threshold: values higher than this get a vertex in the graph
        """
        cosine_similarities = np.empty(shape=(tfidf_matrix.shape[0],
                                              tfidf_matrix.shape[0]))
        degrees = np.zeros(shape=(tfidf_matrix.shape[0]))

        # get cosine similarities between sentences
        for i, sent1 in enumerate(tfidf_matrix):
            for j, sent2 in enumerate(tfidf_matrix):
                calculated_cosine_similarity = cosine_similarity(sent1, sent2)
                if calculated_cosine_similarity > similarity_threshold:
                    cosine_similarities[i, j] = calculated_cosine_similarity
                    degrees[i] += 1
                else:
                    cosine_similarities[i, j] = 0

        cosine_similarities = cosine_similarities / degrees
        return cosine_similarities, degrees


def LexRank(text,percent):
    corpus = article_contents

    porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    punkt = PunktSentenceTokenizer()
    lancaster = LancasterStemmer()

    tfidf_vect = TfidfVectorizer()
    tfidf_matrix, sentences = tokenize_docs(corpus, snowball)
    cosine_similarities, degrees = calculate_similarity_mat(tfidf_matrix)
    lexrank_vector = calc_power_method(cosine_similarities, degrees)

    summmary = ""
    print( [sentences[x] for x in
           np.argsort(lexrank_vector)[::-1][:int(percent)]])
    return summary



    def calc_power_method(similarity_matrix,
                          degrees,
                          stopping_criterion=0.0005,
                          max_loops=1000):

        p_initial = np.ones(shape=len(degrees)) / len(degrees)
        i = 0
        # loop until no change between successive matrix iterations
        while True:
            i += 1
            p_update = np.matmul(cosine_similarities.T, p_initial)
            delta = np.linalg.norm(p_update - p_initial)
            if delta < stopping_criterion or i >= max_loops:
                break
            else:
                p_initial = p_update
        p_update = p_update / np.max(p_update)
        return p_update






























































































































































































































































































































































































































































































































































































































def lex_rank(text_content,percent):
    # TextRank Summarization using Gensim Library.
    # Split is false, gensim return strings joined by "\n". if true, gensim will return list

    summary = summarize(text_content, ratio=(int(percent) / 100), split=False).replace("\n", " ")

    # Returning NLTK Summarization Output
    return summary