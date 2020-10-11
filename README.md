## Introduction

This repository has source codes for learning medical concept embeddings (MCEs) in the following [paper](https://www.medrxiv.org/content/medrxiv/early/2020/07/17/2020.07.14.20151274.full.pdf):

    Comparative Effectiveness of Medical Concept Embedding for Feature Engineering in Phenotyping
        Junghwan Lee*, Cong Liu*, Jae Hyun Kim, Alex Butler, Ning Shang,
        Chao Pang, Karthik Natarajan, Parick Ryan, Casey Ta, Chunhua Weng
        Preprint


### Learning Medical Concept Embeddings
We use [GloVe](https://nlp.stanford.edu/pubs/glove.pdf), [skip-gram](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf), [node2vec](https://snap.stanford.edu/node2vec/), 
[LINE](https://dl.acm.org/doi/abs/10.1145/2736277.2741093?casa_token=Z-RxSgBYo_wAAAAA:pHIx3Cvk9BjV6pIPCXg4UXhA2iFFBaP7TjDRkswkcHY4apy8bIVKmrPa5hMO9HW9gPlzUMcgisnn6-0), and singular value decomposition (SVD) for learning MCEs. For implementation of node2vec and LINE, we used [OpenNE](https://github.com/thunlp/OpenNE), which is an open source python toolkit for network embedding. For implementation of singular value decomposition, we used [SciPy](https://www.scipy.org/).

#### Preparing Dataset
Dataset should be prepared as two pickle files, encoded windowed-EHR and the dictionary for encoded medical concepts in the EHR. Example formats of the data are provided in data/.
1. Encoded windowed-EHR: This is a list of windowed-EHR where each window contains multiple medical concepts. For example, [["concept A", "concept B"], ["Concept A", "Concept C", "Concept D"], ...]. No need to deliminate different patients or windows since we only utilize co-occurrence of the concepts in the same window. All concepts must be encoded with corresponding integer.
2. Concept2id: This is a mapping dictionary for encoded medical concepts in the EHR. For example, if we encoded "Concept A" to integer 0 and "Concept B" to 1, concept2id will look like {0 : "Concept A", 1 : "Concept B"}.

#### Learning Medical Concept Embedding using GloVe
1. Install Python 3.5.2 and all packages in the requirements.txt.
2. Prepare dataset.
3. Start training 

       python src/GloVe.py --input_record <"path of the EHR dataset"> --input_concept2id <"path of the concept2id"> --output <"output path"> --dim <"dimensionality of the embedding"> --batch_size <"batch size for training"> --num_epochs <"training epochs"> --learning_rate <"learning rate for optimizer">
    
    You can check descriptions and the default settings of hyperparameters at help.

       python src/Glove.py --help

#### Learning Medical Concept Embedding using skip-gram
1. Install Python 3.5.2 and all packages in the requirements.txt.
2. Prepare dataset.
3. Start training 

       python src/skipgram.py --input_record <"path of the EHR dataset"> --input_concept2id <"path of the concept2id"> --output <"output path"> --dim <"dimensionality of the embedding"> --batch_size <"batch size for training"> --num_epochs <"training epochs"> --learning_rate <"learning rate for optimizer">
    
    You can check descriptions and the default settings of hyperparameters at help.

       python src/skipgram.py --help

### Acknowledgement
The data containing protected health information have been removed from all publicly available materials.