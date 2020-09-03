import pandas as pd
import numpy as np
import pickle
import gensim
import os
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

def build_simmat(emb_matrix):
    return (1 - pairwise_distances(emb_matrix, metric="cosine"))

def calculate_recall(goldstandard_mce_matrix, goldstandard_concept2id, goldstandard_phenotype_dict, k, mode, error_range=0.05):
    unique_phenotype = list(goldstandard_phenotype_dict.keys())
    unique_phenotype = list(np.sort(unique_phenotype))
    simmat = build_simmat(goldstandard_mce_matrix)
    recall_dict = OrderedDict()
    recall_list = []

    for phenotype in unique_phenotype:
        candidate_concept = goldstandard_phenotype_dict[phenotype]
        if len(candidate_concept) > 1:
            recall, ci = _compute_recall(candidate_concept, k, simmat, goldstandard_concept2id, mode=mode, error_range=error_range)
            recall_list.append(recall)
        else:
            recall = "NA"
            ci = "NA"
        recall_dict.update({phenotype : recall})
    recall_dict.update({"average" : np.mean(recall_list)})

    return recall_dict

def _compute_recall(candidate_concept, k, sim_mat, simmat_dict, mode, error_range=0.05):
    candidate_index = set()
    candidate_num = k
    for concept in candidate_concept:
        candidate_index.add(simmat_dict[concept])
    
    if mode == "percent":
        candidate_num = int(np.ceil(len(candidate_concept) * (k / 100)))
    
    recall_list = []
    for concept in candidate_concept:
        retrieved_concepts = set(np.argsort(sim_mat[simmat_dict[concept]])[(-(candidate_num)-1):-1])
        relevants = len(candidate_index.intersection(retrieved_concepts))
        recall_list.append(relevants)
    
    avg_recall = np.average(np.array(recall_list) / len(candidate_concept))
    ci = [np.percentile(np.array(recall_list), error_range/2), np.percentile(np.array(recall_list), 1-(error_range/2))]
    
    return avg_recall, ci

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