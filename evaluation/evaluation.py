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