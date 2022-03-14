import re
import nltk.corpus
from nltk.corpus import stopwords
import pandas as pd
import yake
import spacy
import numpy as np


def clean_text(df:pd.DataFrame, print_flag=False) -> pd.DataFrame:
    """
    Input: 
    df : Pandas dataframe
    print_flag : indicates if the first item should be printed to illustrate the process of preprocessing

    Returns: 
    cleaned_text : text for each row which is normalized and unicode char free
    """
    cleaned_text = []
    for idx, item in df.iterrows():
        normalized_text = item.Todo.lower()  # normalizing text 
        normalized_text = normalized_text.strip()
        normalized_text = re.sub(r"\s+", " ", normalized_text)
        normalized_text = re.sub(r"\n", "", normalized_text)
        slash_free_text = re.sub(r"/", " ", normalized_text)  # removes "/" and substitutes with " "
        amp_free_text = re.sub("&amp", "",slash_free_text)
        unicode_free_text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", amp_free_text)  # substituting unicode symbols
        stop_word_free_text = " ".join([word for word in unicode_free_text.split() if word not in stopwords.words("english")])  # remove stopwords
        number_free_text = re.sub('\d+', '', stop_word_free_text)
        number_free_text = re.sub(r"\s+", " ", number_free_text)
        cleaned_text.append(number_free_text) 

        # provides examples for chosen index 0 for what happens with the text in the process
        if idx == 0 and print_flag==True: 
            print("Start text: \t\t\t", item.Todo)
            print("\nNormalized text: \t\t", normalized_text)
            print("\nRemoved &amp from text: \t\t", amp_free_text)
            print("\nRemoved / from text: \t\t", slash_free_text)
            print("\nRemoved unicode from text: \t", unicode_free_text)
            print("\nRemoved stopwords text: \t", stop_word_free_text)
            print("\nNumber free: \t\t\t", number_free_text)
            #print("\nLemmatized text: \t\t", lemmatized_text)
    return cleaned_text


def split_sentences(df:pd.DataFrame) -> pd.DataFrame:
    
    #The dataframe column "Todo" will be split according to "!", ".", "?"
    #Returns a dataframe where each sentence is accompanied with the text id 
    
    all_splitted_item = []
    item_indizes = []

    for i, item in df.iterrows():
        todo_item = item.Todo
        splitted_item = re.split("\.|!|\?|;", todo_item) #erhalte Liste retour  # gets sentences 

        for s_item in splitted_item:
            all_splitted_item.append(s_item.strip()) #schreibt s_item in neue zeile (als Liste)
            item_indizes.append(i)  #nehmen den "i" index und schreiben ihn dazu

    df_sentences = pd.DataFrame()
    df_sentences["doc_index"] = item_indizes
    df_sentences["Todo"] = all_splitted_item
    df_sentences.head(20)
    return df_sentences


def tokenize(df:pd.DataFrame) -> list:
    token_arr = []
    for _, item in df.iterrows():
        tokens = nltk.word_tokenize(item.cleaned_text)
        token_arr.append(list(set(tokens)))  # unique tokens
    return token_arr


def get_toplist(df:pd.DataFrame, use_yake=False, language = "en", max_ngram_size=2, deduplication_thresold=0.9, deduplication_algo = 'seqm', windowSize = 2, numOfKeywords = 5, spacy_model="en_core_web_sm") -> list:
    """
    Takes a dataframe and applies either yake or spacy with POS tagging to it -> returns the same dataframe with 
    """
    toplist = []

    if use_yake: 
        yake_kw= []
        yake_values = []
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)

        for text in list(df.cleaned_text):
            keywords = custom_kw_extractor.extract_keywords(text)
            keywords_dict = dict(keywords)
            yake_kw.append(list(keywords_dict.keys()))
            yake_values.append(list(keywords_dict.values()))
    
        # df["yake_kw"] = yake_kw
        # df["yake_values"] = yake_values
        toplist = yake_kw
    else:
        nlp = spacy.load(spacy_model)
        spacy_words = []

        for text in list(df.cleaned_text):
            doc = nlp(text)
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.pos_ == "ADJ" or "NOUN" or "VERB"]
            spacy_words.append(tokens)

        # df["spacy_words"] = spacy_words
        toplist = spacy_words
        
    return toplist


def documentembedding(document:str, model) -> np.array:
    count = 0
    e_v = np.zeros(model.vector_size) 
    for item in document:
           count += 1
           e_v += model.wv[item]
    return e_v / (count+1e-15)  # to counter div by 0 