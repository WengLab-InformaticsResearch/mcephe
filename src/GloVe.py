import numpy as np
import tensorflow as tf
from tqdm import tqdm
import itertools
import random
import sys
import os
import pickle
from collections import defaultdict

class GloVe(tf.keras.Model):
    def __init__(self, embedding_dim=256, max_vocab_size=100, scaling_factor=0.75, batch_size=512, learning_rate=0.01):
        super(GloVe, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.scaling_factor = scaling_factor
        self.batch_size = batch_size
        self.vocab_size = 0
        self.concept2id = None
        self.comap = None
        self.comatrix = None
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        self.epoch_loss_avg = []

    def build_dict(self, corpus):
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(corpus)
        self.concept2id = tokenizer.word_index
        self.vocab_size = len(self.concept2id)

    def save_dict(self, save_dir):
        with open(save_dir + "/concept2id.pkl", "wb") as f:
            pickle.dump(self.concept2id, f)
        print("concept2id successfully saved in the savedir")
     
    def fit_to_corpus(self, corpus):
        self.comap = defaultdict(float)
        self.comatrix = np.zeros((self.vocab_size+1, self.vocab_size+1), dtype=np.float64)
        concept2id = self.concept2id

        for i in tqdm(range(len(corpus))):
            patient = corpus[i]
            for p in patient:
                for k in patient:
                    if p != k:
                        self.comap[(p, k)] += 1
        
        for pair, count in self.comap.items():
            self.comatrix[concept2id[pair[0]], concept2id[pair[1]]] = count

    def init_params(self):
        with tf.device("/cpu:0"):
            """must be implemented with cpu-only env since this is sparse updating"""
            self.target_embeddings = tf.Variable(tf.random.uniform([self.vocab_size+1, self.embedding_dim], 0.1, -0.1),
                                                 name="target_embeddings")
            self.context_embeddings = tf.Variable(tf.random.uniform([self.vocab_size+1, self.embedding_dim], 0.1, -0.1),
                                                  name="context_embeddings")
            self.target_biases = tf.Variable(tf.random.uniform([self.vocab_size+1], 0.1, -0.1),
                                             name='target_biases')
            self.context_biases = tf.Variable(tf.random.uniform([self.vocab_size+1], 0.1, -0.1),
                                              name="context_biases")

    def compute_cost(self, x):
        with tf.device("/gpu:0"):
            """x = [target_ind, context_ind, co_occurrence_count]"""
            target_emb = tf.nn.embedding_lookup([self.target_embeddings], x[0])
            context_emb = tf.nn.embedding_lookup([self.context_embeddings], x[1])
            target_bias = tf.nn.embedding_lookup([self.target_biases], x[0])
            context_bias = tf.nn.embedding_lookup([self.context_biases], x[1])

            weight = tf.math.minimum(1.0, tf.math.pow(tf.math.truediv(x[2], tf.cast(self.max_vocab_size, dtype=tf.float32)),
                                                         self.scaling_factor))
        
            emb_product = tf.math.reduce_sum(tf.math.multiply(target_emb, context_emb), axis=1)
            log_cooccurrence = tf.math.log(tf.add(tf.cast(x[2], dtype=tf.float32), 1))
        
            distance_cost = tf.math.square(
                tf.math.add_n([emb_product, target_bias, context_bias, tf.math.negative(log_cooccurrence)]))
               
            batch_cost = tf.math.reduce_sum(tf.multiply(weight, distance_cost))
          
        return batch_cost

    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            cost = self.compute_cost(x)
        return cost, tape.gradient(cost, self.trainable_variables)

    def prepare_batch(self):
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

    def get_embeddings(self):
        self.embeddings = self.target_embeddings + self.context_embeddings
    
    def save_embeddings(self, save_dir, epoch, avg_loss):
        self.get_embeddings()
        np.save(os.path.join(save_dir, "glove_emb_e{:03d}_loss{:.4f}.npy".format(epoch, avg_loss)),
                self.embeddings)
        print("Embedding results have been saved")

    def train_GloVe(self, num_epochs, save_dir):
        i_ids, j_ids, co_occurs = self.prepare_batch()
        total_batch = int(np.ceil(len(i_ids) / self.batch_size))
        cost_avg = tf.keras.metrics.Mean()

        for epoch in range(num_epochs):
            progbar = tf.keras.utils.Progbar(len(i_ids))
            
            for i in range(total_batch):
                i_batch = i_ids[i * self.batch_size : (i+1) * self.batch_size]
                j_batch = j_ids[i * self.batch_size : (i+1) * self.batch_size]
                co_occurs_batch = co_occurs[i * self.batch_size : (i+1) * self.batch_size]
                cost, gradients = self.compute_gradients([i_batch, j_batch, co_occurs_batch])
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                cost_avg(cost) 
                progbar.add(self.batch_size)
                print("Step {}: Loss: {:.4f}".format(self.optimizer.iterations.numpy(), cost))
                
            if (epoch % 1) == 0: 
                avg_loss = cost_avg.result()
                print("Epoch {}: Loss: {:.4f}".format(epoch, avg_loss))
                self.epoch_loss_avg.append(avg_loss)
                    
        self.save_embeddings(save_dir, epoch, avg_loss)