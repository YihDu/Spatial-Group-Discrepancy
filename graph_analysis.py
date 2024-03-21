import networkx as nx
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
'''
def bootstrap_sample(data, num_dist):
    
    n = len(data)
    bootstrap_samples = []
    for _ in range(num_dist):
        indices = np.random.randint(n , size = n)
        sample = data[indices , :]
        bootstrap_samples.append(sample)
    return bootstrap_samples


def KDE(dist , num_samples , bandwidth = 0.1 ):
    kde = KernelDensity(kernel = 'gaussian' , bandwidth = bandwidth)
    kde.fit(dist)
    samples = kde.sample(num_samples)
    return samples
'''

def KDE(dist , num_samples , bandwidths = np.linspace(0.1 , 1.0 , 30) , random_seed = 42):
    np.random.seed(random_seed)
    kde = KernelDensity(kernel='gaussian')
    params = {'bandwidth': bandwidths}
    grid = GridSearchCV(kde, params, cv=5)
    grid.fit(dist)

    kde = grid.best_estimator_
    samples = kde.sample(num_samples)
    return samples

def get_edge_attributes(truth_g , pred_g , bandwidth = 0.8 , num_dist = None , apply_gene_similarity = False , apply_AD_weight = False):
    unique_groups = set()
    for _, node_data in truth_g.nodes(data=True):
        unique_groups.add(node_data['group'])
    for _, node_data in pred_g.nodes(data=True):
        unique_groups.add(node_data['group'])
        
    group_to_onehot = {group: tuple(int(i == group) for i in unique_groups) for group in unique_groups} #
    pred_edges = list(pred_g.edges())
    truth_edges = list(truth_g.edges())
    
    num_samples = len(pred_edges)
    
    dist_pred = []
    dist_truth = []
    
    for edge in pred_edges:
        u , v = edge
        group_u = pred_g.nodes[u]['group']
        group_v = pred_g.nodes[v]['group']

        if edge in pred_g.edges:
                
                
                if group_u != group_v:
                    encoding = np.zeros(len(unique_groups))
                else:
                    encoding = np.array(group_to_onehot[group_u])#
                
                if apply_gene_similarity:
                    weight = pred_g[u][v]['weight']
                    encoding = encoding * weight
                    
                if apply_AD_weight:
                    ad_weight = pred_g[u][v]['ad_weight']
                    encoding = encoding * ad_weight
                    
                dist_pred.append(encoding)
    
    for edge in truth_edges:
        u , v = edge
        group_u = truth_g.nodes[u]['group']
        group_v = truth_g.nodes[v]['group']
        
        if edge in truth_g.edges:
                
                
                if group_u != group_v:
                    encoding = np.zeros(len(unique_groups))
                else:
                    encoding = np.array(group_to_onehot[group_u])
                
                if apply_gene_similarity:
                    weight = truth_g[u][v]['weight']
                    encoding = encoding * weight
                
                if apply_AD_weight:
                    ad_weight = pred_g[u][v]['ad_weight']
                    encoding = encoding * ad_weight
                  
                dist_truth.append(encoding)     
    
    
    dist_pred = np.array(dist_pred)
    dist_truth = np.array(dist_truth)
    
    distributions_pred = []
    distributions_truth = []
    
    # KDE and sampling

    for i in range(num_dist):
        distribution_pred = KDE(dist_pred,num_samples)
        distribution_truth = KDE(dist_truth,num_samples)
        distributions_pred.append(distribution_pred)
        distributions_truth.append(distribution_truth)

    
    # Bootstrap sampling
    # distributions_pred = bootstrap_sample(dist_pred, num_dist)
    # distributions_truth = bootstrap_sample(dist_truth, num_dist)
    
    return distributions_pred , distributions_truth

def get_edge_attributes_group(pred_g , truth_g , bandwidth = 0.8 , num_dist = 20 ,apply_gene_similarity = False , apply_AD_weight = False):
    unique_groups = set()
    for _, node_data in truth_g.nodes(data=True):
        unique_groups.add(node_data['group'])
    for _, node_data in pred_g.nodes(data=True):
        unique_groups.add(node_data['group'])
    
    group_to_onehot = {group: tuple(int(i == group) for i in unique_groups) for group in unique_groups} #
    pred_edges = list(pred_g.edges())
    truth_edges = list(truth_g.edges())
    
    dist_pred_group = {group : [] for group in unique_groups}
    dist_truth_group = {group : [] for group in unique_groups}

    for edge in pred_edges:
        u , v = edge
        group_u = pred_g.nodes[u]['group']
        group_v = pred_g.nodes[v]['group']

        if edge in pred_g.edges:                
            if group_u != group_v:
                encoding = np.zeros(len(unique_groups))
            else:
                encoding = np.array(group_to_onehot[group_u])
            
            if apply_gene_similarity:
                weight = truth_g[u][v]['weight']
                encoding = encoding * weight
            
            if apply_AD_weight:
                ad_weight = pred_g[u][v]['ad_weight']
                encoding = encoding * ad_weight
            
            dist_pred_group[group_u].append(encoding)
            if group_u != group_v:
                dist_pred_group[group_v].append(encoding)

    for edge in truth_edges:
        u , v = edge
        group_u = truth_g.nodes[u]['group']
        group_v = truth_g.nodes[v]['group']
        
        if edge in truth_g.edges:
                
            if group_u != group_v:
                encoding = np.zeros(len(unique_groups))
            else:
                encoding = np.array(group_to_onehot[group_u])
            
            if apply_gene_similarity:
                weight = truth_g[u][v]['weight']
                encoding = encoding * weight
            
            if apply_AD_weight:
                ad_weight = pred_g[u][v]['ad_weight']
                encoding = encoding * ad_weight
            
        
            dist_truth_group[group_u].append(encoding)
            if group_u != group_v:
                dist_truth_group[group_v].append(encoding)
                
    for group in unique_groups:
        dist_pred_group[group] = np.array(dist_pred_group[group])
        dist_truth_group[group] = np.array(dist_truth_group[group])

    KDE_dist_pred = {group: [] for group in unique_groups}
    KDE_dist_truth = {group: [] for group in unique_groups}
    
    for group in unique_groups:
        num_samples = len(dist_pred_group[group])
        if dist_pred_group[group].size > 0: 
            for i in range(num_dist):
                KDE_sample = KDE(dist_pred_group[group], num_samples=num_samples)
                KDE_dist_pred[group].append(KDE_sample)
        
        if dist_truth_group[group].size > 0:  
            for i in range(num_dist):
                KDE_sample = KDE(dist_truth_group[group], num_samples=num_samples)
                KDE_dist_truth[group].append(KDE_sample)
    
    return KDE_dist_pred, KDE_dist_truth