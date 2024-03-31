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
        
        for i, neighbors in enumerate(indices):
            for n in neighbors[1:]:
                graph.add_edge(i, n)

    def calculate_ad_weight(self , is_binary = True):
        if is_binary:
            for edge in self.truth_G.edges():
                u , v = edge
                if self.truth_G.nodes[u]['group'] == 'A' and self.truth_G.nodes[v]['group'] == 'A':
                    self.truth_G[u][v]['ad_weight'] = 0.7
                elif self.truth_G.nodes[u]['group'] == 'Normal' and self.truth_G.nodes[v]['group'] == 'Normal':
                    self.truth_G[u][v]['ad_weight'] = 0.4
                else:
                    self.truth_G[u][v]['ad_weight'] = 1
            
            
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
                            

    def calculate_gene_weight(self):
        
        weight_map = {
            ("A" , "A_1") : 0.9, # 同属于A Similarity
            ('A_1', 'A'): 0.9,
            ("A" , "A_2") : 0.6,
            ('A_2', 'A'): 0.6,
            ('A', 'B'): 0.5,
            ('B', 'A'): 0.5,
            ('A_1' , 'B'): 0.5,
            ('B' , 'A_1'): 0.5,
            ('A_2' , 'B'): 0.5,
            ('B' , 'A_2'): 0.5
        }
        '''
        weight_map = { 
            ("A" , "B"): 0.3, # distance A B更近
            ("B" , "A"): 0.3,
            ("C" , "A"): 0.7,
            ("A" , "C"): 0.7,
            ("B" , "C"): 0.5,
            ("C" , "B"): 0.5
        }
        '''
        for edge in self.truth_G.edges():
            u , v = edge
            group_u = self.truth_G.nodes[u]['group']
            group_v = self.truth_G.nodes[v]['group']
            
            gene_weight = weight_map.get((group_u , group_v) , 0.95) # 同一种用Similarity 0.9
            self.truth_G[u][v]['gene_weight'] = gene_weight
    
    def copy_weights(self):
        for u , v , data in self.truth_G.edges(data = True):
            if self.pred_G.has_edge(u,v):
                for key , value in data.items():
                    self.pred_G[u][v][key] = value
    
        
    def process_graph(self):
        self.build_graph(self.truth_file , self.truth_G)
        self.build_graph(self.pred_file , self.pred_G)

        if self.apply_gene_similarity:
            self.calculate_gene_weight() 
        
        if self.apply_AD_weight:
            self.calculate_ad_weight(is_binary = False)
        
        self.copy_weights()
                
        return self.truth_G , self.pred_G
                        