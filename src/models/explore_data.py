import copy
from collections import defaultdict

import pandas as pd

from sklearn import decomposition
from sklearn.cluster import KMeans, Birch, SpectralClustering
from sklearn.ensemble import RandomForestClassifier

import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


key_kmeans = 'kmeans'
key_birch = 'birch'
key_spectral = 'spectral'
key_colour = 'color'
key_labels = 'labels'
key_random_forest = 'random_forest'


class Exploration(object):
    def __init__(self, raw_df, target, exlusions = []):
        """
        This constructor really just needs to take in the dataframe, and separate the data into features and targets.
        TODO: what else can go here?
        
        :param raw_df: a Pandas DataFrame that contains both the feature and target information of each element in the data
        :param target: a string or a list of strings that is (or are) the headings of the target column(s).data
        :param exlusions: a list of strings that are the heads of columns in the raw_df that shouldn't be considered either 
                          a feature or a target.
        """
        cols = raw_df.columns
        
        if type(target) == str:
            self.target_key = target
            target_cols = [target]
        elif type(target) == list:
            target_cols = target
        else:
            raise ValueError("target needs to be a string or a list of strings that are the columns of the targets.")


        raw_df.drop(exlusions, axis=1, in_place=True)
        self._df_class = raw_df[target_cols] # the data frame which consists solely of the target variables
        self._df_attributes = raw_df.drop(target_cols, axis=1) # the data frame which consists soley of the attributes
        
        self.reset_inputs()
        
    def reset_inputs(self):
        """ This method takes it all back to the start, undoing all the preprocessing etc """
        self.df_class = copy.copy(self._df_class)
        self.df_attributes = copy.copy(self._df_attributes)
        self.df_pca = None
                
    def preprocess(self, scaler, columns = None, **kwargs):
        """
        This method applies a preprocessing function to the data and saves the fit for later use.
        
        TODO: enforce types
        TODO: make the fit work later using the fit and transform methods.
        :param scaler: a function from sklearn.preprocessing
        :param columns: a list of the columns to receive the preprocessing:
        """
        
        columns = columns if columns else self.df_attributes.columns
        
        self.df_attributes[columns] = scaler.fit_transform(self.df_attributes[columns], **kwargs)
            
    def set_class_to_explore(self, key):
            self.target_key = key
        
    def pca(self, n_components):
        """
        Perform PCA on the df_attributes dataframe to n dimensions of the data.
        
        TODO: replace with a more general decomposition.
        
        :param n_components: an integer specifying the number or components to reduce to.
        """
        X = self.df_attributes
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(X)
        X = pca.transform(X)
        self.df_pca = pd.DataFrame(X)
        
    def cluster(self, cluster_algo, reduced = False, **kwargs):
        """
        Perform a clustering on the df_attributes
        """
        if reduced and self.df_pca is None:
            raise AttributeError("Need to perform dimensionality reduction before doing clustering on reduced data.")
        data = self.df_pca if reduced else self.df_attributes
        n_clusters = len(set(self.df_class[self.target_key]))
        cluster_model = cluster_algo(**kwargs)
        self.cluster_results = cluster_model.fit(data)
        
    def pca_scatter_cluster(self, n = 2, algo = key_kmeans, **kwargs):
        self.pca(n)
        self.cluster(algo = algo, **kwargs)
        int_labels = self.cluster_results.labels_
        text_labels = ['Cluster {}'.format(l) for l in int_labels]
        self.df_pca[key_labels] = text_labels
        self.df_pca[key_colour] = int_labels
        if n == 3:
            return scatter_3d(self.df_pca)
        return scatter_2d(self.df_pca)
        
    def pca_scatter_class(self, n = 2):
        self.pca(n)
        class_values = self.df_class[self.target_key]
        text_labels = class_values
        df_colour_dict = dict([(class_label, i) for (i, class_label) in enumerate(set(class_values))])
        class_colours = np.array([df_colour_dict[key] for key in class_values], dtype=int)
        self.df_pca[key_labels] = text_labels
        self.df_pca[key_colour] = class_colours
        if n == 3:
            return scatter_3d(self.df_pca)
        return scatter_2d(self.df_pca)
    
    def compare_class_clusters_violin(self):
        cluster_int_labels = np.array(self.cluster_results.labels_)
        class_values = self.df_class[self.target_key]
        df_colour_dict = dict([(class_label, i) for (i, class_label) in enumerate(set(class_values))])
        class_int_labels = np.array([df_colour_dict[key] for key in class_values], dtype=int)
        df = pd.DataFrame()
        df['Class'] = class_values
        df['Cluster'] = cluster_int_labels
    
        fig = ff.create_violin(df, data_header='Cluster', group_header='Class',
                       height=500, width=800)
        return py.iplot(fig, filename='Multiple Violins')
   
    def compare_class_clusters_scatter(self):
        cluster_int_labels = np.array(self.cluster_results.labels_)
        class_values = self.df_class[self.target_key]
        
        count_dict = defaultdict(lambda : defaultdict(int))
        
        for cluster, class_value in zip(cluster_int_labels, class_values):
            count_dict[cluster][class_value] += 1
            
        df = pd.DataFrame.from_dict(count_dict)
        
        new_df = df.unstack(level=0).reset_index()
        
        x = 'level_0'
        y = 'level_1'

        x,y = new_df[x], new_df[y]
        trace1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        showlegend=True,
        marker=dict(
            size=new_df[0],
            color=new_df[0],
            colorscale='Jet',
            showscale=True,
            line=dict(
                color=new_df[0],
                width=0.5,
                colorscale='Jet',
            ),

            opacity=1.0
        )
        )

        data = [trace1]
        layout = go.Layout(
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0
            ),
            xaxis=go.XAxis(
                ticks="",
                showticklabels=True,
                tickvals=list(range(len(df.index))),
                ticktext=df.index,
                tickmode="array"
            )
        )
        fig = go.Figure(data=data, layout=layout)
        return py.iplot(fig, filename='simple-3d-scatter')
    
    def classify(self, algo, train_fraction = 0.9, **kwargs):
        
        clf = algo(**kwargs)
        
        n_train = int(len(self.df_attributes)*train_fraction)
        x_train = self.df_attributes[:n_train]
        y_train = self.df_class[self.target_key][:n_train]
        
        x_test = self.df_attributes[n_train:]
        y_test = self.df_class[self.target_key][n_train:]        
        
        
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        return score
        
        