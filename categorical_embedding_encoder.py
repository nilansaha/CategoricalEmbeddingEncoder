import os
import numpy as np
import pandas as pd
from keras.utils import np_utils
from tensorflow.keras import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Flatten, Dense, Embedding
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class CategoricalEmbeddingEncoder():
    """
    Encode categorical feature with embeddings 
    """

    def __init__(self, classification = True, feature_name = 'vector'):
        self.classification = classification
        self.feature_name = feature_name
        self.embedding_size = None
        self.embeddings = None
        self.category_encoder = None

    def fit(self, X, y):
        """
        Fit embedding encoder

        Parameters
        ----------
        X: pandas series containing the categorical feature column
        y: pandas series containing the corresponding targets

        Returns
        -------
        self: returns an instance of self
        """
        categories = len(X.unique())
        self.category_encoder = LabelEncoder()
        X = self.category_encoder.fit_transform(X)
        self.embedding_size = min(50, categories//2 + 1)

        if self.classification:
            loss = 'categorical_crossentropy'
            last_units = len(y.unique())
            last_activation = 'softmax'
            class_encoder = LabelEncoder()
            y = class_encoder.fit_transform(y)
            y = np_utils.to_categorical(y)
        else:
            loss = 'mse'
            last_units = 1
            last_activation = 'sigmoid'
            y = np.asarray(y)

        model = Sequential()
        model.add(Embedding(input_dim = categories, output_dim = self.embedding_size, input_length = 1, name = 'embedding_layer'))
        model.add(Flatten())
        model.add(Dense(20, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(last_units, activation = last_activation))
        model.compile(loss = loss, optimizer = "adam")
        model.fit(X, y, epochs = 100, verbose = 0)
        self.embeddings = model.get_layer('embedding_layer').get_weights()[0]

    
    def transform(self, X):
        """
        Transform labels to embeddings

        Parameters
        ----------
        X : pandas series containing the categorical feature column

        Returns
        -------
        feature_frame : pandas dataframe consisting the categories with embeddings
        """
        X = self.category_encoder.transform(X)
        column_names = [self.feature_name + '_' + str(i) for i in range(self.embedding_size)]
        feature_frame = pd.DataFrame(np.take(self.embeddings, X, axis = 0), columns = column_names)
        return feature_frame

    def fit_transform(self, X, y):
        """
        Fit the encoder and transform the labels to embeddings

        Parameters
        ----------
        X : pandas series containing the categorical feature column
        y : pandas series containing the corresponding targets

        Returns
        -------
        feature_frame : pandas dataframe consisting the categories with embeddings
        """
        self.fit(X, y)
        return self.transform(X)