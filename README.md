## Medcal Concept Embedding for Feature Engineering in Phenotyping

This repository includes source codes for learning and evaluating medical concept embeddings (MCEs) in the following [paper](https://scholar.google.com/citations?user=iSx6QrwAAAAJ&hl=en&oi=ao):

    Medical Concept Embedding for Feature Engineering in Phenotyping 
        Junghwan Lee*, Cong Liu*, Jaehyun Kim, Alex Butler, Casey Ta, Ning Shang,
        Chao Pang, Karthik Natarajan, Parick Ryan, Chunhua Weng
        Preprint


### Learning Medical Concept Embeddings
We use [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) and [node2vec](https://snap.stanford.edu/node2vec/) for learning medical concept embeddings.

#### Learning Medical Concept Embedding using GloVe
1. Install Python 3.5.2 and packages in the requirements.txt.
2. Prepare dataset.
3. Start training by using GloVe.py in /src. 

#### Learning Medical Concept Embedding using node2vec
1. Install Python 3.5.2 and clone the [node2vec repository](https://github.com/aditya-grover/node2vec).
2. Prepare dataset.
3. Third item

### Evaluating Learned Medical Concept Embeddings
We evaluate learned embeddings based on the phenotypes from [PheKB](https://www.phekb.org/).

