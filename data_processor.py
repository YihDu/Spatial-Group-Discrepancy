import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


class GraphBuilder:
    def __init__(self , 
                 pred_file , 
                 truth_file ,
                 GeneExpression_file = None,
                 n_neighbors = 6,
                 apply_gene_similarity = False, 
                 apply_AD_weight = False):
        self.pred_file = pred_file
        self.truth_file = truth_file
        self.GeneExpression_file = GeneExpression_file
        self.apply_gene_similarity = apply_gene_similarity
        self.apply_AD_weight = apply_AD_weight
        self.n_neighbors = n_neighbors
        self.pred_G = nx.Graph()
        self.truth_G = nx.Graph()

    
    def build_graph(self , file_path , graph):
        coordinate_data = pd.read_csv(file_path)
        pos = {}
        
        for idx, row in coordinate_data.iterrows():
            graph.add_node(idx, group=row['group'])
            pos[idx] = (row['x'], row['y'])
            
        pos_array = np.array(list(pos.values()))
        nbrs = NearestNeighbors(n_neighbors = self.n_neighbors + 1).fit(pos_array)
        _ , indices = nbrs.kneighbors(pos_array)
            
        for i , neighbors in enumerate(indices):
            for n in neighbors[1:]:
               graph.add_edge(i , n)
               
    def calculate_gene_similarity(self , graph):
        if self.GeneExpression_file:
            GeneExpression_data = pd.read_csv(self.GeneExpression_file , header = 0 , index_col = 0)
            expression_matrix = GeneExpression_data.values
            expression_matrix = expression_matrix.astype(float)
            normalized_data = (expression_matrix - np.mean(expression_matrix , axis = 1 , keepdims = True)) / np.std(expression_matrix , axis = 1 ,keepdims = True)
            pearson_matrix = np.corrcoef(normalized_data)
            
            for edge in graph.edges():
                u , v = edge
                pearson_weight = pearson_matrix[u,v]
                graph.edges[u,v]['weight'] = 1 - pearson_weight

                
        else:
            raise ValueError("To apply gene similarity weight , gene expression data must be loaded.")    

    '''

    def calculate_gene_similarity(self, graph):
        if self.GeneExpression_file:

            GeneExpression_data = pd.read_csv(self.GeneExpression_file, header=0, index_col=0)
            expression_matrix = GeneExpression_data.values
            expression_matrix = expression_matrix.astype(float)
            

            from scipy.spatial.distance import cdist
            euclidean_matrix = cdist(expression_matrix, expression_matrix, 'euclidean')
            

            distance_weight_matrix = 1 / (1 + euclidean_matrix)
            

            for u, v in graph.edges():

                distance_weight = distance_weight_matrix[u, v]
                graph.edges[u, v]['weight'] = distance_weight
        else:
            raise ValueError("To apply gene similarity weight, gene expression data must be loaded.")
    '''



    def calculate_ad_weight(self):
        truth_data = pd.read_csv(self.truth_file)
        pred_data = pd.read_csv(self.pred_file)
        
        FN_count = 0
        total_count = len(truth_data)
        
        for i in range(total_count):
            if truth_data.loc[i , 'group'] == 'A' and pred_data.loc[i , 'group'] == 'Normal':
               FN_count += 1
        
        ad_weight = 0.5 + FN_count / total_count
        
        for edge in self.truth_G.edges():
            u , v = edge
            if self.truth_G.nodes[u]['group'] == 'A' and self.truth_G.nodes[v]['group'] == 'A':
                self.truth_G[u][v]['ad_weight'] = ad_weight
            elif self.truth_G.nodes[u]['group'] == 'Normal' and self.truth_G.nodes[v]['group'] == 'Normal':
                self.truth_G[u][v]['ad_weight'] = 1 - ad_weight
            else:
                self.truth_G[u][v]['ad_weight'] = 1
        
        for edge in self.pred_G.edges():
            u , v = edge
            if self.pred_G.nodes[u]['group'] == 'A' and self.pred_G.nodes[v]['group'] == 'A':
                self.pred_G[u][v]['ad_weight'] = 1 - ad_weight
            elif self.pred_G.nodes[u]['group'] == 'Normal' and self.pred_G.nodes[v]['group'] == 'Normal':
                self.pred_G[u][v]['ad_weight'] = ad_weight
            else:
                self.pred_G[u][v]['ad_weight'] = 1
        
    def process_graph(self):
        self.build_graph(self.truth_file , self.truth_G)
        self.build_graph(self.pred_file , self.pred_G)

        
        if self.apply_gene_similarity:
            self.calculate_gene_similarity(self.truth_G)
            self.calculate_gene_similarity(self.pred_G)
            
        if self.apply_AD_weight:
            self.calculate_ad_weight()
                
        return self.truth_G , self.pred_G
                        