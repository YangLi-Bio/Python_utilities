#################################################################################
#                                                                               #
# Functions to analyze transcriptomics data with the following characteristics: #
# 1. Large scale                                                                #
# 2. To be streamed with UCSC CellBrowser                                       #
# 3. To be streamed with deeplearning models                                    #
#                                                                               #
#################################################################################


# 1. Scanpy_integrate : integrate multiple scRNA-Seq datasets
# 2. Scanpy_preprocess : preprocess each scRNA-Seq dataset following the weblink : 
#    https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
# 3. cal_growth_weight: calculate growth weights using Prescient


script_dir = "/fs/ess/PCON0022/liyang/Python_utilities/Functions"



#################################################################################
#                                                                               #
#        1. Scanpy_integrate : integrate multiple scRNA-Seq datasets            #
#                                                                               #
#################################################################################


# We use the examples in Scanpy

# The following function comes from : 
# https://scanpy-tutorials.readthedocs.io/en/latest/integrating-data-using-ingest.html

# Input :
# 1. adata_ref : an annData representing the reference dataset, e.g., 
#    adata_ref = sc.datasets.pbmc3k_processed()  # this is an earlier version of the dataset from the pbmc3k tutorial
# 2. adata : an annData denoting the second dataset, i.e., PBMC, e.g., adata = sc.datasets.pbmc68k_reduced()
# 3. ref_label : the label of cell clusters in the reference data, e.g., ref_label = 'louvain'
# 4. label : the label of cell clusters in the query data, e.g., label = 'bulk_labels'

# Output : an annData denoting the integrated dataset


def Scanpy_integrate(adata_ref, adata, ref_label, label):
  
  # Load modules
  import scanpy as sc
  import pandas as pd
  import seaborn as sns
  
  
  # I don't know the meanings
  sc.settings.verbosity = 1 # verbosity: errors (0), warnings (1), info (2), hints (3)
  sc.logging.print_versions()
  sc.settings.set_figure_params(dpi = 80, frameon = False, figsize = (3, 3), facecolor = 'white')
  
  
  # Print the basic information
  print('The reference dataset includes ' + len(adata_ref.obs) + 
  ' cells and ' + len(adata_ref.var) + ' genes.\n')
  print('The query dataset includes ' + len(adata.obs) + 
  ' cells and ' + len(adata.var) + ' genes.\n')
  
  
  # Unify the gene symbol sets between reference and query datasets
  var_names = adata_ref.var_names.intersection(adata.var_names)
  print('We use the ' + len(var_names) + ' common genes between the two datasets.\n')
  adata_ref = adata_ref[:, var_names]
  adata = adata[:, var_names]
  
  
  # Dimension reduction and embeddings for the reference dataset
  # The query data has already been processed, i.e., PCA, UMAP, and neighbor graph building
  sc.pp.pca(adata_ref)
  sc.pp.neighbors(adata_ref)
  sc.tl.umap(adata_ref)
  sc.pl.umap(adata_ref, color = ref_label) # visualization
  
  
  # Mapping query dataset to the reference dataset
  sc.tl.ingest(adata, adata_ref, obs = ref_label)
  adata.uns['louvain_colors'] = adata_ref.uns['louvain_colors']  # fix colors
  sc.pl.umap(adata, color = [ref_label, label], wspace = 0.5)


  # By comparing the ‘bulk_labels’ annotation with ‘louvain’, we see that the data has been 
  # reasonably mapped, only the annotation of dendritic cells seems ambiguous and might have been
  # ambiiguous in adata already.
  adata_concat = adata_ref.concatenate(adata, batch_categories = ['ref', 'new'])
  adata_concat.obs.louvain = adata_concat.obs.louvain.astype('category')
  adata_concat.obs.louvain.cat.reorder_categories(adata_ref.obs.louvain.cat.categories, 
  inplace=True)  # fix category ordering
  adata_concat.uns['louvain_colors'] = adata_ref.uns['louvain_colors']  # fix category colors
  sc.pl.umap(adata_concat, color = ['batch', 'louvain'])
  print('There are ' + len(adata_concat.obs) + ' cells and ' + len(adata_concat.var) + 
  " genes after integration.\n")

  
  # Use BBKNN to integrate the two datasets
  sc.tl.pca(adata_concat)
  
  %%time
  sc.external.pp.bbknn(adata_concat, batch_key='batch')  # running bbknn 1.3.6
  
  sc.pl.umap(adata_concat, color=['batch', ref_label])
  
  
  return adata_concat



#################################################################################
#                                                                               #
#       3. cal_growth_weight: calculate growth weights using Prescient          #
#                                                                               #
#################################################################################


def cal_growth_weight(expr_path, meta_path, time = "Time", out_path = "./"):
  
  # Modules
  print("Importing modules ....")
  import prescient.utils
  import numpy as np
  import pandas as pd
  import sklearn
  import umap
  import scipy
  import annoy
  import torch
  import matplotlib.pyplot as plt
  
  
  
  # Load data
  print("Loading gene expression and meta matrices ...")
  expr = pd.read_csv(expr_path, index_col = 0)
  print("The first five lines of gene expression matrix: ")
  expr.head()
  
  metadata = pd.read_csv(meta_path, index_col = 0)
  print("The first five lines of meta matrix: ")
  metadata.head()
  
  
  # Scale normalized expression for PCA
  scaler = sklearn.preprocessing.StandardScaler()
  xs = pd.DataFrame(scaler.fit_transform(expr), index = expr.index, columns = expr.columns)
  pca = sklearn.decomposition.PCA(n_components = 50)
  xp_ = pca.fit_transform(xs)
  
  
  # Computing growth using built-in PRESCIENT commands
  g, g_l = prescient.utils.get_growth_weights(xs, xp_, metadata, tp_col = time, 
                   genes = list(expr.columns), 
                   birth_gst = out_path + "birth_msigdb_kegg.csv",
                   death_gst = out_path + "death_msigdb_kegg.csv",
                   outfile = out_path + "growth_kegg.pt"
                  )
