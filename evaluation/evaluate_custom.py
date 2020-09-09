import pandas as pd
import numpy as np
import pickle
import gensim
import os
import random
from collections import OrderedDict
from scipy.spatial.distance import cosine
from sklearn.metrics import pairwise_distances

def set_phekb_dict(phekb_data_path):
    """set dictionaries that have condition concepts and drug concepts"""
    phekb_dict = load_data(phekb_data_path)
    unique_phekb_concept = list()
    
    unique_phenotype = list(phekb_dict.keys())
        
    for phenotype in unique_phenotype:
            unique_phekb_concept.extend(phekb_dict[phenotype])
            
    return phekb_dict, unique_phekb_concept

def load_data(pklfile):
    f = open(pklfile, "rb")
    mydata = pickle.load(f)
    return mydata

def build_dict(concept_list):
    mydict = dict()

    for idx, concept in enumerate(concept_list):
        mydict[concept] = idx
    
    return mydict

def load_mce(mce_data_path, format, concept2id_path=None):

    if format in ["glove", "skipgram", "svd"]:
        mce_matrix = np.load(mce_data_path)
        if concept2id_path != None:
            concept2id = load_data(concept2id_path)
        else:
            print("concept2id must be provided if data format is npy")

    elif format in ["line", "node2vec"]:
        mce = gensim.models.KeyedVectors.load_word2vec_format(mce_data_path)
        concept2id = build_dict(list(mce.vocab))

        mce_matrix = np.zeros((len(concept2id), mce.vector_size))

        for concept in list(concept2id.keys()):
            mce_matrix[concept2id[concept]] = mce[concept]

    return mce_matrix, concept2id

def set_goldstandard(mce_matrix, mce_concept2id, phekb_dict, unique_phekb_concept):

    intersection_concept = set.intersection(set(mce_concept2id.keys()), set(unique_phekb_concept))

    goldstandard_phenotype_dict = dict()

    for phenotype in list(phekb_dict.keys()):
        intersection = set.intersection(set(phekb_dict[phenotype]), intersection_concept)
        goldstandard_phenotype_dict[phenotype] = list(intersection)

    goldstandard_mce_matrix = np.zeros((len(intersection_concept), mce_matrix.shape[1]))
    goldstandard_concept2id = build_dict(list(intersection_concept))

    for concept in list(intersection_concept):
        goldstandard_mce_matrix[goldstandard_concept2id[concept]] = mce_matrix[mce_concept2id[concept]]

    return goldstandard_mce_matrix, goldstandard_concept2id, goldstandard_phenotype_dict

def set_custom_seed(seed_list, goldstandard_concept2id, goldstandard_mce_matrix):
    summed_vec = np.zeros(goldstandard_mce_matrix.shape[1])

    for concept in seed_list:
        summed_vec += goldstandard_mce_matrix[goldstandard_concept2id[concept]]

    return normalize_vec(summed_vec)

def normalize_vec(vec):
    return vec / np.linalg.norm(vec, ord=1)

def build_id2concept(concept2id):
    id2concept = dict()
    unique_concepts = list(concept2id.keys())

    for concept in unique_concepts:
        id2concept[concept2id[concept]] = concept
    return id2concept

def load_concept_definition(data_dir):
    with open(data_dir, "r") as f:
        body = f.read()
        raw_list = body.split("\n")
        raw_list = raw_list[1:-5]
    
    concept_definition = dict()
    
    for i in tqdm(range(len(raw_list))):
        contents = raw_list[i].split("\t")
        concept_definition[contents[0]] = contents[1]
        
    return concept_definition

def get_topk_seed(seed_emb, goldstandard_mce_matrix, goldstandard_concept2id, goldstandard_phenotype_dict, concept_definition, k)

    unique_phenotype = list(goldstandard_phenotype_dict.keys())
    goldstandard_id2concept = build_id2concept(goldstandard_concept2id)

    index_rank = _elemwise_cossim(seed_emb, goldstandard_mce_matrix)
    nn_index = index_rank.argsort()[-k:]
    
    nn_concepts = []
    for idx, _ in enumerate(nn_index):
        nn_concepts.append(goldstandard_id2concept[nn_index[-(1+idx)]])

    # calculate recall@k based on all PheKB phenotypes
    recall_dict = OrderedDict()

    for phenotype in unique_phenotype:
        phenotype_concepts = goldstandard_phenotype_dict[phenotype]
        relevants = set.intersection(set(nn_concepts), set(phenotype_concepts))
        recall_dict[phenotype] = len(relevants) / len(phenotype_concepts)

    return nn_concepts, recall_dict

def _elemwise_cossim(vec, matrix):
    noms = np.matmul(vec, matrix.transpose())
    denoms = np.sqrt(np.sum(np.multiply(matrix, matrix), axis=1)) * np.sqrt(np.sum(np.multiply(vec, vec)))
    sim_array = noms / denoms
    
    #print("nan in cosine similarity array is converted to 0")
    sim_array = np.nan_to_num(sim_array)
    return sim_array
