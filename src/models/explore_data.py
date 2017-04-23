import copy

class Exploration(object):
    def __init__(self, raw_df, target_cols):
        """
        This constructor really just needs to take in the dataframe, and separate the data into features and targets.
        TODO: what else can go here?
        """
        cols = raw_df.columns

        self._df_class = raw_df[target_cols] # the data frame which consists solely of the target variables
        self._df_attributes = raw_df.drop(target_cols, axis=1) # the data frame which consists soley of the attributes
        
        self.reset_inputs()
        
    def reset_inputs(self):
        self.df_class = copy.copy(self._df_class)
        self.df_attributes = copy.copy(self._df_attributes)
                
    def preprocess_scale(self, scaler, columns = None, **kwargs):
        """TODO: enforce types"""
        columns = columns if columns else self.df_attributes.columns
        self.df_attributes[columns] = scaler.fit_transform(self.df_attributes[columns], **kwargs)
            
    def preprocess_normalise(self, normaliser, columns = None, **kwargs):
        """TODO: enforce types"""
        columns = columns if columns else self.df_attributes.columns
        self.df_attributes[columns] = scaler.fit_transform(self.df_attributes[columns], **kwargs)
            
    def preprocess(self, function, columns = None):
        """TODO: enforce types"""
        columns = columns if columns else self.df_attributes.columns
        self.df_attributes[columns] = function(self.df_attributes[columns], **kwargs)
        
    def set_class_to_explore(self, key):
            self.target_key = key
        
    def pca(self, n):
        X = self.df_attributes
        pca = decomposition.PCA(n_components=n)
        pca.fit(X)
        X = pca.transform(X)
        self.df_pca = pd.DataFrame(X)
        
    def _cluster_kmeans(self, **kwargs):
        n_clusters = len(set(self.df_class[self.target_key]))
        kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(self.df_pca)
        self.cluster_results = kmeans
        
    def _cluster_birch(self, **kwargs):
        n_clusters = len(set(self.df_class[self.target_key]))
        birch = Birch(n_clusters=n_clusters, **kwargs).fit(self.df_pca)
        self.cluster_results = birch
        
    def _cluster_spectral(self, **kwargs):
        n_clusters = len(set(self.df_class[self.target_key]))
        spectral = SpectralClustering(n_clusters=n_clusters, **kwargs).fit(self.df_pca)
        self.cluster_results = spectral
        
    def cluster(self, algo=key_kmeans, **kwargs):
        if algo==key_kmeans:
            self._cluster_kmeans(**kwargs)
        elif algo==key_birch:
            self._cluster_birch(**kwargs)
        elif algo==key_spectral:
            self._cluster_spectral(**kwargs)
        
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
    
   
    
    def classify(self, algo = key_random_forest, train_fraction = 0.9, **kwargs):
        
        classifiers = {
            'kneighbors': KNeighborsClassifier,
            'svc_1': SVC,
            #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
            'dec_tree': DecisionTreeClassifier,
            key_random_forest: RandomForestClassifier,
            'mlp': MLPClassifier,
            'ada_boost': AdaBoostClassifier,
            'guassian_nb': GaussianNB,
            'quadratic_disc': QuadraticDiscriminantAnalysis
             }
            
        clf = classifiers[algo](**kwargs)
        
        n_train = int(len(self.df_attributes)*train_fraction)
        x_train = self.df_attributes[:n_train]
        y_train = self.df_class[self.target_key][:n_train]
        
        x_test = self.df_attributes[n_train:]
        y_test = self.df_class[self.target_key][n_train:]        
        
        
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        return score
        
        
        
def perturb(grid=0.1):
    return np.random.uniform(low=-grid, high=grid)/2