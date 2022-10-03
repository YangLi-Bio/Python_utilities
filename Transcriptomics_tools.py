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
# 4. run_cellrank_time : run CellRank on time-series scRNA-seq data
# 5. vis_cellrank : visualization for cellrank


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


# Input : 
# 1. expr_path : file path of the gene expression matrix
# 2. meta_path : file path of the metadata
# 3. time : column in the metada representing time labels
# 4. out_path : file path to save outputs
# 5. birth_file : file saving the pathways associated with cell cycle
# 6. death_file : file saving the pathways relevant to cell apoptasis


# Output : growth vector


def cal_growth_weight(expr_path, meta_path, time = "month", out_path = "./", birth_file, death_file):
  
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
  
  
  # expr_path = 'normalized_expr.csv'
  # meta_path = 'metadata.csv'
  # time = "month"
  # out_path = "./"
  # birth_file = 'birth_msigdb_kegg.csv'
  # death_file = 'death_msigdb_kegg.csv'
  
  
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
                   birth_gst = birth_file,
                   death_gst = death_file,
                   outfile = out_path + "growth_kegg.pt"
                  )



#################################################################################
#                                                                               #
#     4. run_cellrank_time : run CellRank on time-series scRNA-seq data         #
#                                                                               #
#################################################################################


# Inlput : 
# 1. adata : annData
# 2. dpi_save : dpi to save images
# 3. dpi : dpi dpi to showcase images
# 4. fontsize : font size in figures
# 5. color_map : coloring scheme
# 6. outDir : directory to save images
# 7. time_label : metadata column to represent time points
# 8. time_start : the start time point
# 9. celltype_label : 


def run_cellrank_time(adata, dpi_save = 400, dpi = 80, 
                      fontsize = 10, color_map = "viridis", 
                      outDir = "./", time_label = "month", 
                      time_palette = ['#FFDE00', '#571179', '#29af7f'], 
                      celltype_palette,
                      time_start = 2.5, figsize = (7, 5),
                      celltype_label = "Shane.cell.type", 
                      organism = "mouse", 
                      basis = "X_wnn.umap"):
  
  # Import modules
  import sys
  import matplotlib.pyplot as plt
  import scvelo as scv
  import scanpy as sc
  import cellrank as cr


  # import CellRank kernels and estimators
  from cellrank.external.kernels import WOTKernel
  from cellrank.tl.kernels import ConnectivityKernel
  from cellrank.tl.estimators import GPCCA
  
  
  # set verbosity levels
  cr.settings.verbosity = 2
  scv.settings.verbosity = 3
  
  
  # figure settings
  scv.settings.set_figure_params(
      "scvelo", dpi_save = dpi_save, dpi = dpi, transparent = True, 
      fontsize = fontsize, color_map = color_map
  )
  
  
  # UMAP plot of time points
  adata = adata.raw.to_adata() # set annData as the raw data
  scv.pl.scatter(adata, basis = basis, c = time_label, legend_loc = "right", 
      palette = time_palette, title = '', legend_fontsize = fontsize,
      figsize = figsize,
      save = outDir + "UMAP_time.pdf")
  # To-do: arrange the order of keys in legend
  
  
  # UMAP visualization of cell types
  scv.pl.scatter(adata, basis = basis, c = celltype_label, legend_loc = "on data",
      title = '', legend_fontsize = fontsize, figsize = figsize,
      save = outDir + "UMAP_celltypes.pdf")
      
  # Pre-process the data
  sc.pp.pca(adata)
  sc.pp.neighbors(adata, random_state = 0)
  
  
  # Estimate initial growth rates
  wk = WOTKernel(adata, time_key = time_label)
  wk.compute_initial_growth_rates(organism = organism, key_added = "growth_rate_init")
  scv.pl.scatter(
      adata, c = "growth_rate_init", legend_loc = "right", basis = basis, s = 10, 
      save = outDir + "growth_rate_init.pdf"
  )
  
  
  # Compute transition matrix
  wk.compute_transition_matrix(
    growth_iters = 3, growth_rate_key = "growth_rate_init", last_time_point = "connectivities"
  )
    
  
  return adata, wk


#################################################################################
#                                                                               #
#                     5. vis_cellrank : visualization for cellrank              #
#                                                                               #
#################################################################################


# To-do : add RNA velocity
def vis_cellrank(adata, wk, time_label = "Time", time_start, basis = "X_wnn.umap",
                genes_oi, n_sims = 300, max_i = 200, dpi = 80, outDir = "./", 
                cand_ter_states):
  
  # Simulate random walks
  wk.plot_random_walks(
    n_sims = n_sims,
    max_iter = max_iter,
    start_ixs = {time_label: time_start},
    basis = basis,
    c = time_label,
    legend_loc = "right",
    linealpha = 0.5,
    dpi = dpi * 2,
  )
  # To-do: save this figure
  
  scv.pl.scatter(adata, c = celltype_label, basis = basis, legend_loc = "right", 
                save = outDir + "UMAP_celltypes.pdf")
  
  
  # Probability mass flow in time
  ax = wk.plot_single_flow(
    cluster_key = celltype_label,
    time_key = time_label,
    cluster = "MEF/other",
    min_flow = 0.1,
    xticks_step_size = 4,
    show = False,
    dpi = dpi,
    clusters = ["MEF/other", "MET", "Stromal"]
  )


  # prettify the plot a bit, rotate x-axis tick labels
  locs, labels = plt.xticks()
  ax.set_xticks(locs)
  ax.set_xticklabels(labels, rotation = 90)
  plt.save(outDir + "Mass_flow_time.pdf")
  
  
  # Compute macrostates
  ck = ConnectivityKernel(adata)
  ck.compute_transition_matrix()
  combined_kernel = 0.9 * wk + 0.1 * ck
  g = GPCCA(combined_kernel)
  g.compute_schur()
  g.plot_spectrum(real_only = True)
  # To-do: save the figure
  
  g.compute_macrostates(n_states = 6, cluster_key = celltype_label)
  g.plot_macrostates(discrete = True, basis = basis, legend_loc = "right")
  # To-do: save the figure

  
  # Define terminal macrostates
  g.plot_macrostate_composition(key = time_label)
  # To-do: save the figure
  
  g.set_terminal_states_from_macrostates(cand_ter_states)
  
  
  # Compute fate probabilities
  g.compute_absorption_probabilities(solver = "gmres", use_petsc = True)
  cr.pl.circular_projection(adata, keys = [celltype_label, time_label], 
                            legend_loc = "right", title = "")
  # To-do: save the figure
  
  
  # Log-odds in time
  cr.pl.log_odds(
    adata,
    lineage_1 = "IPS",
    lineage_2 = None,
    time_key = time_label,
    keys = ["Obox6"],
    threshold = 0,
    size = 2,
    xticks_step_size = 4,
    figsize = (9, 4)
  )
  cr.pl.log_odds(
    adata,
    lineage_1 = "IPS",
    lineage_2 = None,
    time_key = time_label,
    keys = [celltype_label],
    threshold = 0,
    size = 2,
    xticks_step_size = 4,
    figsize = (9, 4),
    legend_loc = "upper right out"
  )
  
  
  # Driver genes
  g.compute_lineage_drivers(return_drivers = False)
  
  
  return adata, wk, ax, g
