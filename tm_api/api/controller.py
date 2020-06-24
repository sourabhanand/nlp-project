import os
from flask import Blueprint, request, jsonify, render_template, json, session,\
    redirect, url_for, flash
import sys
print(sys.path)
from models.lda import do_lda
from datautils.utils import get_processed_data

extraction_app = Blueprint('extraction_app', __name__)
DATA_LOC = os.path.join('data', '2020-06-20')


@extraction_app.route('/', methods=['GET'])
def get_home_page():
    return "You have entered Awesome Topic Modelling!!"


@extraction_app.route('/lda', methods=['GET'])
def get_similar_docs():
    data_df = get_processed_data(DATA_LOC)
    do_lda(data_df)
    return "You have entered Awesome Topic Modelling!!"


