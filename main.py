#from data_processor import GraphBuilder
from data_processor_CaseStudy import GraphBuilder # for toy example in the cases
from graph_analysis import get_edge_attributes , get_edge_attributes_group
from SGD_calculator import *
import networkx as nx
import time
from GK_emd import gaussian_emd


def main(coordinate_file_truth , 
         coordinate_file_pred , 
         gene_expr_file = None , 
         apply_gene_similarity = False,
         apply_AD_weight = False,
         is_subtying = False):

     start_time = time.time()
    
     graph_builder = GraphBuilder(
          pred_file = coordinate_file_pred , 
          truth_file = coordinate_file_truth ,
          GeneExpression_file  = gene_expr_file , 
          apply_gene_similarity = apply_gene_similarity, 
          apply_AD_weight = apply_AD_weight
          )

     graph_truth , graph_pred = graph_builder.process_graph()
     
     graph_building_time = time.time()
     print(f"Graph Building took {graph_building_time - start_time:.2f} seconds.")
     
     if not is_subtying:

          samples_truth , samples_pred = get_edge_attributes(graph_truth , 
                                                            graph_pred , 
                                                            num_dist = 50 , 
                                                            apply_gene_similarity = apply_gene_similarity , 
                                                            apply_AD_weight = apply_AD_weight)
          attribute_end_time = time.time()
          print(f"Getting edge attributes took {attribute_end_time - graph_building_time:.2f} seconds.")
          
          
          SGD_score = compute_mmd(samples_truth , samples_pred ,kernel = gaussian_emd,is_hist = True)

          
          return SGD_score
     
     if is_subtying:
          
          samples_truth , samples_pred = get_edge_attributes_group(graph_truth , 
                                                            graph_pred , 
                                                            bandwidth = 0.5 ,
                                                            num_dist = 30 , 
                                                            apply_gene_similarity = apply_gene_similarity , 
                                                            apply_AD_weight = apply_AD_weight)
          SGD_score_1 =  compute_mmd(samples_truth['A'] , samples_pred['A'] ,kernel = gaussian_emd,is_hist = True)
          SGD_score_2 =  compute_mmd(samples_truth['B'] , samples_pred['B'] ,kernel = gaussian_emd,is_hist = True)
          SGD_score_3 =  compute_mmd(samples_truth['C'] , samples_pred['C'] ,kernel = gaussian_emd,is_hist = True)

          return SGD_score_1 , SGD_score_2 , SGD_score_3

if __name__ == "__main__":
     'test'
     coordinate_file_truth = "data/simulate data/Case5/truth_dissim_SpatialData.csv"
     coordinate_file_pred = "data/simulate data/Case5/pred13_SpatialData.csv"

     gene_expr_file = "data/simulate data/Case5/dissim_GeneExpressionData.csv"
     
     SGD = main(coordinate_file_truth , 
        coordinate_file_pred , 
        gene_expr_file , 
        apply_gene_similarity = True , 
        apply_AD_weight = False , 
        is_subtying = False)

     print(SGD)     