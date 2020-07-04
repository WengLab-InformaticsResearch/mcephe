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

        self.similarity_matrix = dict()
        self.simmat_dict = dict()

        self.glove_concept2id = OrderedDict()
        self.glove_emb_matrix = OrderedDict()
        
        self.n2v_recall = OrderedDict()
        self.glove_recall = OrderedDict()
        self.n2v_precision = OrderedDict()
        self.glove_precision = OrderedDict()

    def _loadGloveModel(self):
        print("load MCEs trained by GloVe...")
        self.concept2id_fiveyearemb = load_dictionary(self.config.results.concept2id_fiveyearemb)
        self.concept2id_visitemb = load_dictionary(self.config.results.concept2id_visitemb)
        self.fiveyearemb = np.load(self.config.results.fiveyearemb)
        self.visitemb = np.load(self.config.results.visitemb)
        print("MCEs trained by GloVe have been loaded")

    def _loadn2vModel(self):
        print("load MCEs trained by node2vec...")
        self.graphemb_model = gensim.models.KeyedVectors.load_word2vec_format(self.config.results.graphemb)
        self.graphembplus_model = gensim.models.KeyedVectors.load_word2vec_format(self.config.results.graphembplus)
        print("MCEs trained by node2vec have been loaded")

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
        print("build similarity matrix for MCEs...")
        self._buildSimilarityMatrix_n2vmodel()
        self._buildSimilarityMatrix_glovemodel()

    def _buildSimilarityMatrix_glovemodel(self):
        unique_concept = count_unique(self.concept_phe)
        fiveyearemb_concept = set(self.concept2id_fiveyearemb.keys())
        visitemb_concept = set(self.concept2id_visitemb.keys())
        fiveyearemb_intersection_concept = set.intersection(unique_concept, fiveyearemb_concept)
        visitemb_intersection_concept = set.intersection(unique_concept, visitemb_concept)

        fiveyearemb_dict = dict()
        visitemb_dict = dict()

        for concept in list(fiveyearemb_intersection_concept):
            fiveyearemb_dict[concept] = self.fiveyearemb[self.concept2id_fiveyearemb[concept]]
        
        for concept in list(visitemb_intersection_concept):
            visitemb_dict[concept] = self.visitemb[self.concept2id_visitemb[concept]]

        fiveyearemb_matrix = np.zeros((len(fiveyearemb_dict.keys()), self.config.hparams.emb_dim))
        visitemb_matrix = np.zeros((len(visitemb_dict.keys()), self.config.hparams.emb_dim))

        fiveyearemb_simmat_dict = build_newdict(list(fiveyearemb_dict.keys()))
        visitemb_simmat_dict = build_newdict(list(visitemb_dict.keys()))

        for concept in list(fiveyearemb_simmat_dict.keys()):
            fiveyearemb_matrix[fiveyearemb_simmat_dict[concept]] = fiveyearemb_dict[concept]
    
        for concept in list(visitemb_simmat_dict.keys()):
            visitemb_matrix[visitemb_simmat_dict[concept]] = visitemb_dict[concept]

        fiveyearemb_simmat = 1 - pairwise_distances(fiveyearemb_matrix, metric="cosine")
        visitemb_simmat = 1 - pairwise_distances(visitemb_matrix, metric="cosine")

        self.similarity_matrix["fiveyearemb_simmat"] = fiveyearemb_simmat
        self.similarity_matrix["visitemb_simmat"] = visitemb_simmat

        self.simmat_dict["fiveyearemb_simmat_dict"] = fiveyearemb_simmat_dict
        self.simmat_dict["visitemb_simmat_dict"] = visitemb_simmat_dict

    def _buildSimilarityMatrix_n2vmodel(self):
        unique_concept = count_unique(self.concept_phe)
        graphemb_concept = set(self.graphemb_model.vocab)
        graphembplus_concept = set(self.graphembplus_model.vocab)
        graphemb_intersection_concept = set.intersection(unique_concept, graphemb_concept)
        graphembplus_intersection_concept = set.intersection(unique_concept, graphembplus_concept)

        graphemb_dict = dict()
        graphembplus_dict = dict()
    
        for concept in list(graphemb_intersection_concept):
            graphemb_dict[concept] = self.graphemb_model[concept]
        
        for concept in list(graphembplus_intersection_concept):
            graphembplus_dict[concept] = self.graphembplus_model[concept]
        
        graphemb_matrix = np.zeros((len(graphemb_dict.keys()), self.config.hparams.emb_dim))
        graphembplus_matrix = np.zeros((len(graphembplus_dict.keys()), self.config.hparams.emb_dim))
    
        graphemb_simmat_dict = build_newdict(list(graphemb_dict.keys()))
        graphembplus_simmat_dict = build_newdict(list(graphembplus_dict.keys()))
    
        for concept in list(graphemb_simmat_dict.keys()):
            graphemb_matrix[graphemb_simmat_dict[concept]] = graphemb_dict[concept]
    
        for concept in list(graphembplus_simmat_dict.keys()):
            graphembplus_matrix[graphembplus_simmat_dict[concept]] = graphembplus_dict[concept]
        
        graphemb_simmat = 1 - pairwise_distances(graphemb_matrix, metric="cosine")
        graphembplus_simmat = 1 - pairwise_distances(graphembplus_matrix, metric="cosine")

        self.similarity_matrix["graphemb_simmat"] = graphemb_simmat
        self.similarity_matrix["graphembplus_simmat"] = graphembplus_simmat

        self.simmat_dict["graphemb_simmat_dict"] = graphemb_simmat_dict
        self.simmat_dict["graphembplus_simmat_dict"] = graphembplus_simmat_dict

    def computePrecision(self, k, mode):
        pass

    def computeRecall(self, k, mode):
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

def count_unique(my_dict):
    unique_concept = []
    for key in list(my_dict.keys()):
        unique_concept.extend(my_dict[key])
    return unique_concept

def build_newdict(concept_list):
    mydict = dict()
    for i in range(len(concept_list)):
        mydict[concept_list[i]] = i
    return mydict

