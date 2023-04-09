from pprint import pprint
import textwrap
import json
import requests
import string
import re
import nltk
import string
import itertools
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
import pke
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import traceback
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import streamlit as st
import random

st.title(':blue[Fill in the Blanks Question Generator]')
st.subheader("Enter your text")
text = st.text_area(' ',height=200)
st.button('Submit')

wrapper = textwrap.TextWrapper(width = 150)
word_list = wrapper.wrap(text = text)
#for element in word_list:
  #print(element)
# function for Tokenize sentence

def tokenize_sentences(text):
  sentences = sent_tokenize(text)
  sentences = [sentence.strip() for sentence in sentences if len(sentence) > 50]
  return sentences

sentences = tokenize_sentences(text)
#print(sentences)

# Function for Get NOUN VERB ADJECTIVE
def get_noun_adj_verb(text):
  out = []

  try:
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input = text, language = 'en')

    pos = {'VERB', 'ADJ', 'NOUN'}
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')


    extractor.candidate_selection(pos = pos)
    extractor.candidate_weighting(alpha = 1.1,
                                  threshold = 0.75,
                                  method = 'average')
    
    keyphrases = extractor.get_n_best(n=30)

    for val in keyphrases:
      out.append(val[0])

  except:
    out = []
    traceback.print_exc()

  return out

noun_verbs_adj = get_noun_adj_verb(text)
#print ("keywords: ",noun_verbs_adj)

# Function the sentences where Keywords is used
def get_sentences_for_keywords(keywords, sentences):

  keyword_processor = KeywordProcessor()
  keyword_sentences = {}

  for word in keywords:
    keyword_sentences[word] = []
    keyword_processor.add_keyword(word)

  for sentence in sentences:
    keywords_found = keyword_processor.extract_keywords(sentence)
    for key in keywords_found:
      keyword_sentences[key].append(sentence)

  for key in keyword_sentences.keys():
    values = keyword_sentences[key]
    values = sorted(values, key = len, reverse=True)
    keyword_sentences[key] = values

  return keyword_sentences

keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keywords(noun_verbs_adj, sentences)
#pprint (keyword_sentence_mapping_noun_verbs_adj)

# Fill in the blanks
def get_fill_in_the_blanks(sentence_mapping):
  out = {}
  blank_sentences = []
  processed = []
  keys = []

  for key in sentence_mapping:
    if len(sentence_mapping[key])>0:
      sent = sentence_mapping[key][0]

      insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
      no_of_replacements = len(re.findall(re.escape(key), sent , re.IGNORECASE))
      line = insensitive_sent.sub(' ______________ ', sent)

      if (sentence_mapping[key][0] not in processed) and no_of_replacements<2:
        blank_sentences.append(line)
        processed.append(sentence_mapping[key][0])
        keys.append(key)

  out['Sentence'] = blank_sentences[:10]
  out['keys'] = keys[:10]
  return out

fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping_noun_verbs_adj)

count = 0
count1 = 0
sent = fill_in_the_blanks['Sentence']
st.subheader(':blue[Sentences are:]')
for i in sent:
  count = count+1
  st.latex(str(count)+'.'+str(i))
  

key1 =fill_in_the_blanks['keys']
random.shuffle(key1)

st.subheader(':green[Options are:]')
for x in key1:
  count1 = count1+1
  st.text(str(count1)+'.'+str(x))

