## Medcal Concept Embedding for Feature Engineering in Phenotyping

This repository includes source codes for learning and evaluating medical concept embeddings (MCEs) in the following [paper](https://scholar.google.com/citations?user=iSx6QrwAAAAJ&hl=en&oi=ao):

    Medical Concept Embedding for Feature Engineering in Phenotyping 
        Junghwan Lee*, Cong Liu*, Jaehyun Kim, Alex Butler, Ning Shang,
        Chao Pang, Karthik Natarajan, Parick Ryan, Casey Ta, Chunhua Weng
        Preprint


### Learning Medical Concept Embeddings
We use [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) and [node2vec](https://snap.stanford.edu/node2vec/) for learning medical concept embeddings.

#### Learning Medical Concept Embedding using GloVe
1. Install Python 3.5.2 and all packages in the requirements.txt.
2. Prepare dataset as a pickle file. GloVe takes a list of patient visits, where each visit of a patient contains multiple medical concepts, as an input. For example, [["concept A", "concept B"], ["Concept A", "Concept C", "Concept D"], ...]. No need to deliminate different patients since GloVe calculates co-occurrence based on a single visit. Concepts do not need to be encoded as integer. The GloVe package automatically encodes all existing unique codes in the training data and will output a mapping dictionary between concepts and corresponding integer codes.
3. Start training 

       python src/GloVe.py --input <"input path"> --output <"output path">
    
    You can check descriptions of hyperparameters and the default settings at help.

       python src/Glove.py --help

#### Learning Medical Concept Embedding using node2vec
1. Install Python 3.5.2 and clone the [node2vec repository](https://github.com/aditya-grover/node2vec).
2. Prepare dataset. node2vec takes a list of relationships between concepts as an input. Two concepts in a relationship must be deliminated by a space and different relationships must be deliminated by a linebreak like the following:

       <"concept_id_1"> <"concept_id_2">
       <"concept_id_1"> <"concept_id_3">

    An example input format is provided at the node2vec repository. Note that if relationships have a direction, order of concepts must be aligned according to the direction. Concepts are required to be encoded as integers.

3. Start training

       python main.py --input <"path of input data"> --output <"path to save results">

    Detailed instruction is provided at the node2vec repository.


### Evaluating Learned Medical Concept Embeddings
We evaluate learned embeddings based on the phenotypes from [PheKB](https://www.phekb.org/). Condition (i.e. diagnosis) concepts included in 33 phenotypes were provided in /data/concepts_phenotypes.pkl as a pickle file. All concepts were converted to [OMOP CDM](https://www.ohdsi.org/data-standardization/the-common-data-model/). 

1. Train MCEs to be evaluated.
2. Prepare a master json file that contains hyperparameters and the default settings. This json file should include information about the locations of MCEs, directories, and hyperparameters. An example format of the json file is provided at /data.
3. Start evaluating

       python evaluation/evaluation.py --input <"The path of master json file"> --k <"k in precision and recall"> --mode <"choice between @k or @k%">

### Acknowledgement
The data containing protected health information (PHI) have been removed from all publicly available materials.
