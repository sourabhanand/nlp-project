###############################################################################
# CS5803 NLP MiniProject - Exploration to Covid 19
# CS18MDS11007 - Sourabh Anand
# CS18MDS11008 - Gourang Gaurav
#
# -*- coding: utf-8 -*-
###############################################################################

import os
import json
import gensim
import pandas as pd
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

DATA_LOC = os.path.join('..','data','2020-06-07')
# union of stop words from both gensim and nltk
STOP_WORDS = set(STOPWORDS) | set(stopwords.words('english'))

def parse_json(cord_uid, json_file):
  """parse json files to get abstract text and body text
     Returns a list [corr_uid, abstract_text, paper_text]
     This can be stored in dataframe to optimize reading 
     of all files again and again
  """
  row_data = [cord_uid]
  try:
    with open(json_file) as jf:
      data = json.load(jf)
    abstracts = data['abstract']
    body_texts = data['body_text']
    abstract_text = ''
    paper_text = ''
    for abstract in abstracts:
      abstract_text = abstract_text + ' ' + abstract['text']
    for body_text in body_texts:
      paper_text = paper_text + ' ' + body_text['text']

    row_data.append(abstract_text)
    row_data.append(paper_text)
  except:
    pass
  return row_data

def lemmatize(pos_tokens):
  """lemmatize words based on pos tag"""
  lemmatizer = WordNetLemmatizer()
  for word, tag in pos_tokens:
    if tag.startswith("NN"):
      yield lemmatizer.lemmatize(word, pos='n')
    elif tag.startswith('VB'):
      yield lemmatizer.lemmatize(word, pos='v')
    elif tag.startswith('JJ'):
      yield lemmatizer.lemmatize(word, pos='a')
    else:
      yield word

def preprocess_data(paper_text):
  """Tokenize, remove stopwords and lemmatize"""
  tokens = simple_preprocess(paper_text, deacc=True)
  tokens = [t for t in tokens if t not in STOP_WORDS]
  pos_tokens = pos_tag(tokens)
  print(len(list(lemmatize(pos_tokens))))

def main():
  metadata_df = pd.read_csv(os.path.join(DATA_LOC,'metadata.csv'), low_memory=False)

  read_all_data = False
  if read_all_data:
    # filter rows which contains parsed json files (either pdf_json or pmc_json)
    pdf_mdata_df = metadata_df[metadata_df['pdf_json_files'].notnull()
                   & metadata_df['pdf_json_files'].str.startswith('document_parses')]
    pmc_mdata_df = metadata_df[metadata_df['pmc_json_files'].notnull()
                   & metadata_df['pmc_json_files'].str.startswith('document_parses')]
    parsed_mdata_df = pdf_mdata_df.merge(pmc_mdata_df, how='outer')
    num_rows = len(parsed_mdata_df.index)
    print("No of rows", num_rows)

    # iterate rows of dataframe and read the json files
    data_df = pd.DataFrame(index=range(num_rows), columns=['cord_uid', 'abstract', 'paper_text'])
    print(data_df.head())
    row_idx = 0
    for row in parsed_mdata_df.itertuples():
      json_files_str = row.pmc_json_files if pd.isnull(row.pdf_json_files) else row.pdf_json_files
      json_files = json_files_str.split(';')
      for json_file in json_files:
        print('Row:', row_idx)
        json_file = os.path.join(DATA_LOC, json_file.strip())
        row_data = parse_json(row.cord_uid, json_file)
        if len(row_data) == 3:
          data_df.loc[row_idx] = row_data
          row_idx += 1

    print('Writing to CSV')
    data_df.to_csv('final_data.csv')
  else:
    print('Loading processed data')
    data_df = pd.read_csv('final_data.csv')
    data_df.drop(['Unnamed: 0'], axis=1, inplace=True)

  # preprocess the text
  # This function can be 'applied' to data frame column containing paper_text or abstract_text
  preprocess_data(data_df.iloc[0]['paper_text'])


if __name__ == '__main__' : main()
