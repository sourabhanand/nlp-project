import os
from models.lda import doLDA
from datautils.utils import get_processed_data

DATA_LOC = os.path.join('data','2020-06-07')

def main():
  data_df = get_processed_data(DATA_LOC)
  doLDA(data_df)

if __name__ == '__main__' : main()
