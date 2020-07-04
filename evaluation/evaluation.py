import pandas as pd
from collections import OrderedDict
import pickle
import itertools
import json
from tqdm import tqdm
import numpy as np
from dotmap import DotMap
import gensim
import tensorflow as tf
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import os

class EvaluateMCE(object):
    """Class for evaluation of MCEs"""
    def __init__(self, json_dir):
        self.config = set_config(json_dir)
        
        self.concept_phe = load_dictionary(self.config.data.phe_dict)

        self.graphemb_phedict = OrderedDict()
        self.graphembplus_phedict = OrderedDict()
        self.fiveyearemb_phedict = OrderedDict()
        self.visitemb_phedict = OrderedDict()
        
        self.graphemb_model = None
        self.graphembplus_model = None
        self.concept2id_5yremb = None
        self.concept2id_visitemb = None

        self.glove_concept2id = OrderedDict()
        self.glove_emb_matrix = OrderedDict()
        
        self.n2v_simmat = OrderedDict()
        self.glove_simmat = OrderedDict()
        
        self.n2v_simmat_dict = OrderedDict()
        self.glove_simmat_dict = OrderedDict()
        
        self.n2v_recall = OrderedDict()
        self.glove_recall = OrderedDict()
        self.n2v_precision = OrderedDict()
        self.glove_precision = OrderedDict()

    def _loadGloveModel(self):
        self.concept2id_fiveyearemb = load_dictionary(self.config.results.concept2id_fiveyearemb)
        self.concept2id_visitemb = load_dictionary(self.config.results.concept2id_visitemb)
        print("GloVe models have been loaded")

    def _loadn2vModel(self):
        print("load node2vec model...")
        self.graphemb_model = gensim.models.KeyedVectors.load_word2vec_format(self.config.results.graphemb)
        self.graphembplus_model = gensim.models.KeyedVectors.load_word2vec_format(self.config.results.graphembplus)
        print("node2vec models have been loaded")

    def setPheDict(self):
        self._loadn2vModel()
        self._loadGloveModel()

        graphemb_concept = set(self.graphemb_model.vocab)
        graphembplus_concept = set(self.graphembplus_model.vocab)
        fiveyearemb_concept = set(self.concept2id_fiveyearemb.keys())
        visitemb_concept = set(self.concept2id_visitemb.keys())
        unique_phenotype = list(self.concept_phe.keys())

        for phe in unique_phenotype:
            intersection_concept = graphemb_concept.intersection(set(self.concept_phe[phe]))
            self.graphemb_phedict.update({phe : list(intersection_concept)})

        for phe in unique_phenotype:
            intersection_concept = graphembplus_concept.intersection(set(self.concept_phe[phe]))
            self.graphembplus_phedict.update({phe : list(intersection_concept)})

        for phe in unique_phenotype:
            intersection_concept = fiveyearemb_concept.intersection(set(self.concept_phe[phe]))
            self.fiveyearemb_phedict.update({phe : list(intersection_concept)})

        for phe in unique_phenotype:
            intersection_concept = visitemb_concept.intersection(set(self.concept_phe[phe]))
            self.visitemb_phedict.update({phe : list(intersection_concept)})

    def buildSimilarityMatrix(self):
        pass

    def computePrecision(self):
        pass

    def computeRecall(self):
        pass
        
    def saveResults(self):
        pass

def set_config(json_file):
    """
    Get config data from a json file
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        json_body = json.load(config_file)
    # convert the dictionary to a namespace using bunch lib
    config = DotMap(json_body)
    return config       

def load_dictionary(data_dir):
    with open(data_dir, 'rb') as f:
        my_dict = pickle.load(f)
    return my_dict