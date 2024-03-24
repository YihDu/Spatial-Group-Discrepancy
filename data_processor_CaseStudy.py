import pandas as pd
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


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
        label_map = {}
        
        for idx, row in coordinate_data.iterrows():
            graph.add_node(idx, group=row['group'])
            pos[idx] = (row['x'], row['y'])
            label_map[idx] = row['group']
            
        pos_array = np.array(list(pos.values()))
        nbrs = NearestNeighbors(n_neighbors = self.n_neighbors + 1).fit(pos_array)
        _ , indices = nbrs.kneighbors(pos_array)
        
        if self.apply_gene_similarity:
            for i , neighbors in enumerate(indices):
                for n in neighbors[1:]:
                    if graph is self.truth_G:
                        weight = self.calculate_weight_toyexample(label_map[i] , label_map[n])
                        graph.add_edge(i , n , weight = weight)
                    else:
                        graph.add_edge(i , n)
            if graph is self.pred_G:
                self.copy_weight_from_truth()
        else:
            for i , neighbors in enumerate(indices):
                for n in neighbors[1:]:
                    graph.add_edge(i , n)
    
    def calculate_weight_toyexample(self , label_u , label_v):
        if label_u == label_v:
            return 0.9 # similarity
        elif (label_u == "A" and label_v == "A_1") or (label_u == "A_1" and label_v == "A"): # A A1 没那么相似 similarity
            return 0.7
        elif (label_u == "A" and label_v == "B") or (label_u == "B" and label_v == "A"): # B1 B2
            return 0.5 # distance
        elif (label_u == "A_1" and label_v == "B") or (label_u == "B" and label_v == "A_1"):
            return 0.5 # distance 暂时设计成一致
        else:
            return 1.0
            
    def copy_weight_from_truth(self):
        for u , v in self.truth_G.edges():
            if self.pred_G.has_edge(u,v):
                self.pred_G.edges[u,v]["weight"] = self.truth_G.edges[u,v]["weight"]
            
    
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



    def calculate_ad_weight(self , is_binary = True):
        truth_data = pd.read_csv(self.truth_file)
        pred_data = pd.read_csv(self.pred_file)
        
        FN_count = 0
        total_count = len(truth_data)
        
        for i in range(total_count):
            if truth_data.loc[i , 'group'] == 'A' and pred_data.loc[i , 'group'] == 'Normal':
               FN_count += 1
        
        #ad_weight = 0.5 + FN_count / total_count
        ad_weight = 0.8
        
        if is_binary:
            for edge in self.truth_G.edges():
                u , v = edge
                if self.truth_G.nodes[u]['group'] == 'A' and self.truth_G.nodes[v]['group'] == 'A':
                    self.truth_G[u][v]['ad_weight'] = 0.8
                elif self.truth_G.nodes[u]['group'] == 'Normal' and self.truth_G.nodes[v]['group'] == 'Normal':
                    self.truth_G[u][v]['ad_weight'] = 0.2
                else:
                    self.truth_G[u][v]['ad_weight'] = 1
            
            for edge in self.pred_G.edges():
                u , v = edge
                if self.pred_G.nodes[u]['group'] == 'A' and self.pred_G.nodes[v]['group'] == 'A':
                    self.pred_G[u][v]['ad_weight'] = 0.2
                elif self.pred_G.nodes[u]['group'] == 'Normal' and self.pred_G.nodes[v]['group'] == 'Normal':
                    self.pred_G[u][v]['ad_weight'] = 0.8
                else:
                    self.pred_G[u][v]['ad_weight'] = 1
            
        if not is_binary:
            for edge in self.truth_G.edges():
                u , v = edge
                if self.truth_G.nodes[u]['group'] == 'B' and self.truth_G.nodes[v]['group'] == 'B':
                    self.truth_G[u][v]['ad_weight'] = 0.9
                elif self.truth_G.nodes[u]['group'] == 'C' and self.truth_G.nodes[v]['group'] == 'C':
                    self.truth_G[u][v]['ad_weight'] = 0.6
                elif self.truth_G.nodes[u]['group'] == 'A' and self.truth_G.nodes[v]['group'] == 'A':
                    self.truth_G[u][v]['ad_weight'] = 0.1
                else:
                    self.truth_G[u][v]['ad_weight'] = 1   
            
            for edge in self.pred_G.edges():
                u , v = edge
                if self.pred_G.nodes[u]['group'] == 'B' and self.pred_G.nodes[v]['group'] == 'B':
                    self.pred_G[u][v]['ad_weight'] = 0.9
                elif self.pred_G.nodes[u]['group'] == 'C' and self.pred_G.nodes[v]['group'] == 'C':
                    self.pred_G[u][v]['ad_weight'] = 0.6
                elif self.pred_G.nodes[u]['group'] == 'A' and self.pred_G.nodes[v]['group'] == 'A':
                    self.pred_G[u][v]['ad_weight'] = 0.1
                else:
                    self.pred_G[u][v]['ad_weight'] = 1                    
        
    def process_graph(self):
        self.build_graph(self.truth_file , self.truth_G)
        self.build_graph(self.pred_file , self.pred_G)

        
        if self.apply_AD_weight:
            self.calculate_ad_weight(is_binary = False)
                
        return self.truth_G , self.pred_G
                        