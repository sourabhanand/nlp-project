import os
from models.lda import do_lda
from datautils.utils import get_processed_data

DATA_LOC = os.path.join('data', '2020-06-20')


def main():
    data_df = get_processed_data(DATA_LOC, use_cached=True)
    do_lda(data_df)


if __name__ == '__main__':
    main()
