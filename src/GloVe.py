import numpy as np
import tensorflow as tf
from tqdm import tqdm
import itertools
import random
import os
import pickle
import argparse
from collections import defaultdict

class GloVe(tf.keras.Model):
    def __init__(self, save_dir, embedding_dim=256, max_vocab_size=1000, scaling_factor=0.75, batch_size=512, num_epochs=50, learning_rate=0.01):
        super(GloVe, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.scaling_factor = scaling_factor
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.vocab_size = 0
        self.concept2id = None
        self.comap = None
        self.comatrix = None
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        self.save_dir = save_dir
        self.epoch_loss_avg = []
     
    def buildCoMatrix(self, patient_record):
        self.concept2id = build_dict(count_unique(patient_record))
        self.vocab_size = len(self.concept2id.keys())
        self.comap = defaultdict(float)
        self.comatrix = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)

        for i in tqdm(range(len(patient_record))):
            patient = patient_record[i]
            for p in patient:
                for k in patient:
                    if p != k:
                        self.comap[(p, k)] += 1
        
        for pair, count in self.comap.items():
            self.comatrix[self.concept2id[pair[0]], self.concept2id[pair[1]]] = count

    def initParams(self):
        with tf.device("/cpu:0"):
            """must be implemented with cpu-only env since this is sparse updating"""
            self.target_embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_dim], 0.1, -0.1),
                                                 name="target_embeddings")
            self.context_embeddings = tf.Variable(tf.random.uniform([self.vocab_size, self.embedding_dim], 0.1, -0.1),
                                                  name="context_embeddings")
            self.target_biases = tf.Variable(tf.random.uniform([self.vocab_size], 0.1, -0.1),
                                             name='target_biases')
            self.context_biases = tf.Variable(tf.random.uniform([self.vocab_size], 0.1, -0.1),
                                              name="context_biases")

    def computeCost(self, x):
        with tf.device("/gpu:0"):
            """x = [target_ind, context_ind, co_occurrence_count]"""
            target_emb = tf.nn.embedding_lookup([self.target_embeddings], x[0])
            context_emb = tf.nn.embedding_lookup([self.context_embeddings], x[1])
            target_bias = tf.nn.embedding_lookup([self.target_biases], x[0])
            context_bias = tf.nn.embedding_lookup([self.context_biases], x[1])

            weight = tf.math.minimum(1.0, tf.math.pow(tf.math.truediv(x[2], tf.cast(self.max_vocab_size, dtype=tf.float32)), self.scaling_factor))
        
            emb_product = tf.math.reduce_sum(tf.math.multiply(target_emb, context_emb), axis=1)
            log_cooccurrence = tf.math.log(tf.add(tf.cast(x[2], dtype=tf.float32), 1))
        
            distance_cost = tf.math.square(
                tf.math.add_n([emb_product, target_bias, context_bias, tf.math.negative(log_cooccurrence)]))
               
            batch_cost = tf.math.reduce_sum(tf.multiply(weight, distance_cost))
          
        return batch_cost

    def computeGradients(self, x):
        with tf.GradientTape() as tape:
            cost = self.computeCost(x)
        return cost, tape.gradient(cost, self.trainable_variables)

    def prepareBatch(self):
        i_ids = []
        j_ids = []
        co_occurs = []
        comap_list = list(self.comap.items())

        for pair, co_occur in comap_list:
            i_ids.append(self.concept2id[pair[0]])
            j_ids.append(self.concept2id[pair[1]])
            co_occurs.append(co_occur)
     
        assert len(i_ids) == len(j_ids), "The length of the data are not the same"
        assert len(i_ids) == len(co_occurs), "The length of the data are not the same"
        return i_ids, j_ids, co_occurs

    def getEmbeddings(self):
        self.embeddings = self.target_embeddings + self.context_embeddings
    
    def saveEmbeddings(self, epoch, avg_loss):
        self.getEmbeddings()
        np.save(os.path.join(self.save_dir, "glove_emb_e{:03d}_loss{:.4f}.npy".format(epoch, avg_loss)),
                self.embeddings)
        print("Embedding results have been saved in the output path")

    def saveConcept2id(self):
        with open(self.save_dir + "/concept2id_glove.pkl", "wb") as f:
            pickle.dump(self.concept2id, f)
        print("concept2id successfully saved in the output path")

    def trainModel(self):
        i_ids, j_ids, co_occurs = self.prepareBatch()
        total_batch = int(np.ceil(len(i_ids) / self.batch_size))
        cost_avg = tf.keras.metrics.Mean()

        for epoch in range(self.num_epochs):
            progbar = tf.keras.utils.Progbar(len(i_ids))
            
            for i in range(total_batch):
                i_batch = i_ids[i * self.batch_size : (i+1) * self.batch_size]
                j_batch = j_ids[i * self.batch_size : (i+1) * self.batch_size]
                co_occurs_batch = co_occurs[i * self.batch_size : (i+1) * self.batch_size]
                cost, gradients = self.computeGradients([i_batch, j_batch, co_occurs_batch])
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost_avg(cost) 
                progbar.add(self.batch_size)
                print("Step {}: Loss: {:.4f}".format(self.optimizer.iterations.numpy(), cost))
                
            if (epoch % 1) == 0: 
                avg_loss = cost_avg.result()
                print("Epoch {}: Loss: {:.4f}".format(epoch, avg_loss))
                self.epoch_loss_avg.append(avg_loss)
                    
        self.saveEmbeddings(epoch, avg_loss)
        self.saveConcept2id()

def load_data(data_dir):
    with open(data_dir, 'rb') as f:
        my_data = pickle.load(f)
    return my_data

def count_unique(patient_record):
    """count unique concpets in the patient record"""
    concept_list = []
    for visit in patient_record:
        concept_list.extend(visit)
    return list(set(concept_list))

def build_dict(patient_record):
    unique_concept = count_unique(patient_record)
    my_dict = dict()

    for i in range(len(unique_concept)):
        my_dict[unique_concept[i]] = i

    return my_dict

def parse_arguments(parser):
    parser.add_argument("--input", type=str, help="The path of training data")
    parser.add_argument("--output", type=str, help="The path to output results")
    parser.add_argument("--dim", type=int, default=200, help="The dimension of embeddings")
    parser.add_argument("--max_vocab", type=int, default=1000, help="The maximum vocabulary size")
    parser.add_argument("--scaling_factor", type=float, default=0.75, help="The scaling factor")
    parser.add_argument("--batch_size", type=int, default=512000, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for Adagrad optimizer")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    training_data = load_data(args.data_dir)
    GloVe_model = GloVe(args.output, args.dim, args.max_vocab, args.scaling_factor, args.batch_size, args.num_epochs, args.learning_rate)
    GloVe_model.buildCoMatrix(training_data)
    GloVe_model.initParams()
    GloVe_model.trainModel()