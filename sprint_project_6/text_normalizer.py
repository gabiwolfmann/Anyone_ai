import re
import nltk
import spacy
from unidecode import unidecode
from bs4 import BeautifulSoup
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from tqdm.notebook import tqdm
nltk.download('stopwords')

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = spacy.load('en_core_web_sm')

porter = nltk.stem.PorterStemmer()


def remove_html_tags(text):
    # Put your code
    soup = BeautifulSoup(text,'html.parser') 
    text = soup.get_text()
    return text


def stem_text(text,ToktokTokenizer=True):
    # Put your code
    if ToktokTokenizer:
        token_words=tokenizer.tokenize(text)
    else:
        token_words=nltk.word_tokenize(text)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
    text = " ".join(stem_sentence)
    return text


def lemmatize_text(text):
    # Put your code
    doc=nlp(text)
    text=" ".join([token.lemma_ for token in doc])
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # Put your code
    for contraction, word in contraction_mapping.items():
        text = text.replace(contraction, word)
    return text


def remove_accented_chars(text):
    # Put your code
    text=unidecode(text)
    return text


def remove_special_chars(text, remove_digits=False):
    # Put your code
    if remove_digits:
        text = re.sub(r"[^a-zA-Z ]", "", text)
    else:
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list, spacy_stopwords=False):
    # Put your code
    words = nltk.word_tokenize(text)
    if spacy_stopwords:
        stopwords = nlp.Defaults.stop_words
    without_stop_words = [word for word in words if not word.lower() in stopwords]
    text =" ".join([word for word in without_stop_words])
    return text


def remove_extra_new_lines(text):
    # Put your code
    text = text.replace("\n", " ")
    return text


def remove_extra_whitespace(text):
    # Put your code
    text = " ".join(text.split())
    return text
    

def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list,
    ToktokTokenizer=True,
    spacy_stopwords=False

):
    
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in tqdm(corpus):
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)
            
        # Remove extra newlines
        doc = remove_extra_new_lines(doc)
        
        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)
            
        # Expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
            
        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
            
        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc,ToktokTokenizer=ToktokTokenizer)
            
        # Remove special chars and\or digits    
        if special_char_removal:
            doc = remove_special_chars(
                doc,
                remove_digits=remove_digits
            )  

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

         # Lowercase the text    
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc,
                is_lower_case=text_lower_case,
                stopwords=stopwords,
                spacy_stopwords=spacy_stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
