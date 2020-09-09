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

def set_seed(k, goldstandard_phenotype_dict, goldstandard_concept2id):
    """set k seed for phenotypes"""
    seed_dict = OrderedDict()
    unique_phenotype = list(goldstandard_phenotype_dict.keys())

    for phenotype in unique_phenotype:
        seed_emb = []
        if len(goldstandard_phenotype_dict[phenotype]) < 10:
            seed_dict[phenotype] = "NA"
        else:
            for i in range(len(goldstandard_phenotype_dict[phenotype])):
                seed = random.sample(goldstandard_phenotype_dict[phenotype], k)
                seed_emb.append(_sum_seed(seed, goldstandard_concept2id, goldstandard_mce_matrix))
            seed_dict[phenotype] = seed_emb
    
    return seed_dict

def _sum_seed(seed, goldstandard_concept2id, goldstandard_mce_matrix):
    summed_vec = np.zeros(goldstandard_mce_matrix.shape[1])

    for concept in seed:
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

def calculate_recall(seed_dict, goldstandard_mce_matrix, goldstandard_concept2id, goldstandard_phenotype_dict, k, mode):
    unique_phenotype = list(seed_dict.keys())

    recall_dict = OrderedDict()
    recall_list = []
    goldstandard_id2concept = build_id2concept(goldstandard_concept2id)

    for phenotype in unique_phenotype:
        seed_emb = seed_dict[phenotype]
        if seed_emb != "NA":
            recall = _compute_recall(seed_emb, goldstandard_phenotype_dict[phenotype], goldstandard_mce_matrix, goldstandard_id2concept, k, mode)
            recall_list.append(recall)
        else:
            recall = "NA"
        recall_dict.update({phenotype : recall})
    recall_dict.update({"average" : np.mean(recall_list)})

    return recall_dict

def _compute_recall(seed_emb, phenotype_concepts, mce_matrix, id2concept, k, mode):

    candidate_num = k
    if mode == "percent":
        candidate_num = int(np.ceil(len(phenotype_concepts) * (k / 100)))
        
    recall_list = []
    for i in range(len(seed_emb)):
        index_rank = _elemwise_cossim(seed_emb[i], mce_matrix)
        nn_index = index_rank.argsort()[-candidate_num:]
    
        nn_concepts = []
        for ind in nn_index:
            nn_concepts.append(id2concept[ind])

        relevants = set.intersection(set(nn_concepts), set(phenotype_concepts))
        recall_list.append(len(relevants) / len(phenotype_concepts))
    
    return np.mean(recall_list)

def _elemwise_cossim(vec, matrix):
    noms = np.matmul(vec, matrix.transpose())
    denoms = np.sqrt(np.sum(np.multiply(matrix, matrix), axis=1)) * np.sqrt(np.sum(np.multiply(vec, vec)))
    sim_array = noms / denoms
    
    #print("nan in cosine similarity array is converted to 0")
    sim_array = np.nan_to_num(sim_array)
    return sim_array

def parse_arguments(parser):
    parser.add_argument("--input_mce", type=str, help="The path of mce results")
    parser.add_argument("--input_phekb", type=str, help="The path of Phekb data")
    parser.add_argument("--input_concept2id", type=str, default=None, help="The path of concept2id")
    parser.add_argument("--mce_format", type=str, help="The format of the mce result")
    parser.add_argument("--output", type=str, help="The path to output results")
    parser.add_argument("--k", type=int, help="Recall@k")
    parser.add_argument("--mode", type=str, help="num indicates traditional top-k recall and percent indicates recall@k%")  args = parser.parse_args()
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    phekb_dict, unique_phekb_concept = set_phekb_dict(args.input_phekb)
    mce_matrix, mce_concept2id = load_mce(args.input_mce, args.mce_format, concept2id_path=args.input_concept2id)
    goldstandard_mce_matrix, goldstandard_concept2id, goldstandard_phenotype_dict = set_goldstandard(mce_matrix, mce_concept2id, phekb_dict, unique_phekb_concept)
    recall_dict = calculate_recall(goldstandard_mce_matrix, goldstandard_concept2id, goldstandard_phenotype_dict, args.k, args.mode, error_range=0.05)
    
    recall_df = pd.DataFrame(recall_dict)
    recall_df.to_csv(os.path.join(args.output, 
    "{format}_{k}_{mode}_recall.csv".format(format=args.mce_format, k=args.k, mode=args.mode)))
    print("computed recall was saved in the output path...")