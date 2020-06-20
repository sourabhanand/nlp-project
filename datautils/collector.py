###############################################################################
# CS5803 NLP MiniProject - Exploration to Covid 19
# CS18MDS11007 - Sourabh Anand
# CS18MDS11008 - Gourang Gaurav
#
# -*- coding: utf-8 -*-
###############################################################################

import os
import json
import pandas as pd

DATA_LOC = os.path.join('..','data','2020-06-07')

def main():
  metadata_df = pd.read_csv(os.path.join(DATA_LOC,'metadata.csv'), low_memory=False)

  # filter rows which contains parsed json files (either pdf_json or pmc_json)
  pdf_mdata_df = metadata_df[metadata_df['pdf_json_files'].notnull()
                 & metadata_df['pdf_json_files'].str.startswith('document_parses')]
  pmc_mdata_df = metadata_df[metadata_df['pmc_json_files'].notnull()
                 & metadata_df['pmc_json_files'].str.startswith('document_parses')]
  parsed_mdata_df = pdf_mdata_df.merge(pmc_mdata_df, how='outer')
  print("No of rows", len(parsed_mdata_df.index))

  # iterate rows of dataframe and read the json files
  for row in parsed_mdata_df.itertuples():
    json_files_str = row.pmc_json_files if pd.isnull(row.pdf_json_files) else row.pdf_json_files
    json_files = json_files_str.split(';')
    for json_file in json_files:
      json_file = os.path.join(DATA_LOC, json_file.strip())


if __name__ == '__main__' : main()
