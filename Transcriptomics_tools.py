#################################################################################
#                                                                               #
# Functions to analyze transcriptomics data with the following characteristics  #
#                                                                               #
#################################################################################


# 1. Scanpy_integrate : integrate multiple scRNA-Seq datasets
# 2. Scanpy_preprocess : preprocess each scRNA-Seq dataset following the weblink : 
#    https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
# 3. cal_growth_weight: calculate growth weights using Prescient
# 4. run_cellrank_time : run CellRank on time-series scRNA-seq data
# 5. vis_cellrank : visualization for cellrank
# 6. cellrank_pseudotime : run cellrank in a pseudotime scenario
# 7. scanpy_export_CB : export scRNA-seq dataset to Cell Browser
# 8. scanpy_annotate : annotate cell clusters
# 9. scanpy_trajectory : trajectory analysis using Scanpy
# 10. scanpy_pt : pseudotime analysis using Scanpy
# 11. pyscenic_grn : infer gene regulatory networks using pySCENIC


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
  
  sc.external.pp.bbknn(adata_concat, batch_key='batch')  # running bbknn 1.3.6
  
  sc.pl.umap(adata_concat, color=['batch', ref_label])
  
  
  return adata_concat



#################################################################################
#                                                                               #
# 2. Scanpy_preprocess_clustering : preprocess each scRNA-Seq dataset following #
#    the weblink : https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.htm #                                                                               #
#                                                                               #
#################################################################################


# Input :
# 1. adata : annData
# 2. dir : directory to save images or orther results
# 3. n_top : the number of top-ranked highly expressed genes, 20 by default


def Scanpy_preprocess_clustering(adata, key_genes, outDir = "./", n_top = 20, min_genes = 200, 
                                 min_cells = 3, n_genes_by_counts = 2500, n_genes = 25,
                                 pct_counts_mt = 5, deg_method = "t-test"):
  
  # Modules and setting
  import numpy as np
  import pandas as pd
  import scanpy as sc
  
  sc.settings.verbosity = 3 # verbosity: errors (0), warnings (1), info (2), hints (3)
  sc.logging.print_header()
  sc.settings.set_figure_params(dpi = 80, facecolor = 'white')
  adata.var_names_make_unique() # this is unnecessary if using `var_names='gene_ids'` 
  # in `sc.read_10x_mtx`
  
  
  # Data description
  print ("Basic information of the annData: ")
  adata
  
  
  # Preprocessing
  print ("highest fraction of counts in each single cell, across all cells, showing the " + 
          n_top + " genes:")
  sc.pl.highest_expr_genes(adata, n_top = n_top, save = outDir + "highest_expr_genes.png")
  
  
  # Basic filtering
  print ("Filtering genes expressed in " + min_cells + " cells and filtering cells where at least " + 
          min_genes + " are expressed:")
  sc.pp.filter_cells(adata, min_genes = min_genes)
  sc.pp.filter_genes(adata, min_cells = min_cells)
  print ("Adding MT-genes as annotations ...")
  adata.var['mt'] = adata.var_names.str.startswith('MT-')  
  # annotate the group of mitochondrial genes as 'mt'
  
  sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
  print ("Violin plots for quality control: ")
  sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter = 0.4, multi_panel = True, save = outDir + "violin_plots.png")
  sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
  print ("Violin plots for quality control: ")
  sc.pl.scatter(adata, x = 'total_counts', y = 'n_genes_by_counts', 
                save = outDir + "scatter_plots.png")
  print ("slicing the AnnData object by retaining cells with at least " + 
          n_genes_by_counts + " genes and at most " + pct_counts_mt + " MT genes ...")
  adata = adata[adata.obs.n_genes_by_counts < n_genes_by_counts, :]
  adata = adata[adata.obs.pct_counts_mt < pct_counts_mt, :]
  print ("Normalizing the data matrix to 1e4 reads per cell ...")
  sc.pp.normalize_total(adata, target_sum=1e4)
  sc.pp.log1p(adata)
  sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
  print ("Highly variable genes: ")
  sc.pl.highly_variable_genes(adata, save = outDir + "highly_variable_genes.png")
  adata.raw = adata
  print ("Retaining the highly variable genes ...")
  adata = adata[:, adata.var.highly_variable]
  print ("Regressing out effects of total counts per cell and the percentage of MT-genes ...")
  sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
  print ("Scaling each gene to unit variance. Clipping values exceeding standard deviation 10 ...")
  sc.pp.scale(adata, max_value=10)
  
  
  # Principal component analysis
  print ("Reducing dimensions ...")
  sc.tl.pca(adata, svd_solver='arpack')
  print ("Making a scatter plot in the PCA coordinate: ")
  sc.pl.pca(adata, color='CST3', save = outDir + "PCA_scatter_plot.png")
  print ("PCA variance ratio: ")
  sc.pl.pca_variance_ratio(adata, log=True, save = outDir + "variance_ratio.png")
  
  
  # Computing the neighborhood graph
  print ("Computing neighborhood graph ...")
  sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
  
  
  # Embedding the neighborhood graph
  print ("Embedding the neighborhood graph ...")
  sc.tl.paga(adata)
  sc.pl.paga(adata, plot=False) # remove `plot=False` if you want to see the coarse-grained graph
  sc.tl.umap(adata, init_pos='paga', save = outDir + "UMAP.png")
  print ("Raw count UMAP of genes: ")
  sc.pl.umap(adata, color= key_genes, save = outDir + "UMAP_gene_rawCt.png")
  print ("Scaled count UMAP of genes: ")
  sc.pl.umap(adata, color= key_genes, use_raw = False, save = outDir + "UMAP_gene_scaledCt.png")


  # Clustering the neighborhood graph
  print ("Clustering the neighborhood graph ...")
  sc.tl.leiden(adata)
  sc.pl.umap(adata, color= "leiden", save = outDir + "cluster_UMAP.png")
  
  
  # Finding marker genes
  print ("Finding marker genes using " + method + " ...")
  if deg_method == "t-test":
      sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
  elif deg_method == "wilcoxon":
      sc.settings.verbosity = 2  # reduce the verbosity
      sc.tl.rank_genes_groups(adata, 'leiden', method= deg_method)
  sc.pl.rank_genes_groups(adata, n_genes=n_genes, sharey=False, save = outDir + "markers.png")
  print ("The top-5 top-ranked genes per cluster: ")
  pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
  
  
  # Get a table with the scores and groups
  result = adata.uns['rank_genes_groups']
  groups = result['names'].dtype.names
  pd.DataFrame(
      {group + '_' + key[:1]: result[key][group]
      for group in groups for key in ['names', 'pvals', 'pvals_adj']}).head(5)
  
  
  return adata
  


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


def cal_growth_weight(expr_path, meta_path, birth_file, death_file, time = "month", out_path = "./"):
  
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
  
  
  return g, g_l



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


def run_cellrank_time(adata, celltype_palette,
                      dpi_save = 400, dpi = 80, 
                      fontsize = 10, color_map = "viridis", 
                      outDir = "./", time_label = "month", 
                      time_palette = ['#FFDE00', '#571179', '#29af7f'], 
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
  # To-do : refer to materials regarding unbalanced optimal transport
  adata.obs[time_label] = adata.obs[time_label].astype('float')
  wk = WOTKernel(adata, time_key = time_label)
  wk.compute_initial_growth_rates(organism = organism, key_added = "growth_rate_init")
  scv.pl.scatter(
      adata, c = "growth_rate_init", legend_loc = "right", basis = basis, s = 10, 
      figsize = figsize, title = '', legend_fontsize = fontsize,
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
def vis_cellrank(adata, wk, cand_ter_states, genes_oi, time_start, time_label = "Time", basis = "X_wnn.umap",
                n_sims = 300, max_iter = 200, dpi = 80, outDir = "./", 
                figsize = (7, 5), cluster_in = "OPC", clusters_out = ["Oligo", "AG", "MG"]
                ):
  
  # Simulate random walks
  wk.plot_random_walks(
    n_sims = n_sims,
    max_iter = max_iter,
    start_ixs = {time_label: time_start},
    basis = basis,
    c = time_label,
    legend_loc = "right",
    linealpha = 0.5,
    dpi = dpi * 2, figsize = figsize,
    save = outDir + "cellrank_random_walk.pdf"
  )

  
  # UMAP plot of cell types
  scv.pl.scatter(adata, basis = basis, c = celltype_label, legend_loc = "on data",
      title = '', legend_fontsize = fontsize, figsize = figsize,
      save = outDir + "UMAP_celltypes.pdf")
  
  
  # Probability mass flow in time
  ax = wk.plot_single_flow(
    cluster_key = celltype_label,
    time_key = time_label,
    cluster = cluster_in,
    min_flow = 0.1,
    xticks_step_size = 4,
    show = False,
    dpi = dpi,
    clusters = clusters_out
  )
  # To-do: only one cell type, maybe because of too few cell types.


  # prettify the plot a bit, rotate x-axis tick labels
  locs, labels = plt.xticks()
  ax.set_xticks(locs)
  ax.set_xticklabels(labels, rotation = 90)
  plt.savefig(outDir + "Mass_flow_time.pdf")
  
  
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



#################################################################################
#                                                                               #
#         6. cellrank_pseudotime : run cellrank in a pseudotime scenario        #
#                                                                               #
#################################################################################


# Input : 
# 1. adata : annData
# 2. cluster_key : used to get PAGA graph
# 3. n_genes : number of driver genes to identify


def cellrank_pseudotime(adata, path = "./", prefix = "tmp", 
                        cluster_key = "clusters", weight_connectivities=0.2, 
                        n_genes = 5):
  
  # Import modules
  print ("Importing modules ...\n")
  import sys
  import scvelo as scv
  import scanpy as sc
  import cellrank as cr
  import numpy as np
  scv.settings.verbosity = 3
  scv.settings.set_figure_params("scvelo")
  cr.settings.verbosity = 2
  
  
  # Data description
  print("Loading annData object: \n")
  print(adata)
  
  
  import warnings
  warnings.simplefilter("ignore", category=UserWarning)
  warnings.simplefilter("ignore", category=FutureWarning)
  warnings.simplefilter("ignore", category=DeprecationWarning)
  
  
  # Evaluate the data
  print("Showing the fraction of spliced/unspliced reads ...")
  scv.pl.proportions(adata, save = path + prefix + "_splicing_prop.png")
  
  
  # Preprocess the data
  print("Peprocessing the annData ...\n")
  # print("Filtering out genes with less than 2000 spliced/unspliced counts ...\n")
  # print("Normalizing and log-transforming the data restricted to 2000 highly variable genes ...\n")
  scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
  # print("Performing dimension reduction ...\n")
  sc.tl.pca(adata)
  # print("Selecting the top-30 neighbors based on 30 PCs ...\n")
  sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
  scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
  
  
  # Run scVCelo
  print ("Running scVCelo ...\n")
  scv.tl.recover_dynamics(adata, n_jobs=8)
  scv.tl.velocity(adata, mode="dynamical")
  scv.tl.velocity_graph(adata)
  scv.pl.velocity_embedding_stream(
    adata, basis="umap", legend_fontsize=12, title="", smooth=0.8, min_mass=4, 
    save = path + prefix + "_velocity.png"
  )
  
  
  # Identify terminal states
  print ("Identifying the terminal state ...\n")
  cr.tl.terminal_states(adata, cluster_key=cluster_key, weight_connectivities=weight_connectivities)
  cr.pl.terminal_states(adata, save = path + prefix + "_terminal_states.png")
  
  
  # Identify initial states
  print ("Identifying the initial state ...\n")
  cr.tl.initial_states(adata, cluster_key=cluster_key)
  cr.pl.initial_states(adata, discrete=True, save = path + prefix + "_initial_states.png")
  
  
  # Compute fate maps
  cr.tl.lineages(adata)
  cr.pl.lineages(adata, same_plot=False, save = path + prefix + "_lineages.png")
  cr.pl.lineages(adata, same_plot=True, save = path + prefix + "_cell_fates.png")
  
  
  # Directed PAGA
  print ("Generating directed PAGA ...\n")
  scv.tl.recover_latent_time(
    adata, root_key="initial_states_probs", end_key="terminal_states_probs"
  )
  scv.tl.paga(
    adata,
    groups=cluster_key,
    root_key="initial_states_probs",
    end_key="terminal_states_probs",
    use_time_prior="velocity_pseudotime",
  )
  cr.pl.cluster_fates(
    adata,
    mode="paga_pie",
    cluster_key=cluster_key,
    basis="umap",
    legend_kwargs={"loc": "top right out"},
    legend_loc="top left out",
    node_size_scale=5,
    edge_width_scale=1,
    max_edge_width=4,
    title="directed PAGA",
    save = path + prefix + "_PAGA.png"
  )
  
  
  # Compute lineage drivers
  print ("Computing lineage drivers ...\n")
  cr.tl.lineage_drivers(adata)
  cr.pl.lineage_drivers(adata, lineage="Alpha", n_genes=n_genes, 
    save = path + prefix + "_drivers.png")
  
  
  print ("Returning the annData ...\n")
  return adata



#################################################################################
#                                                                               #
#        7. scanpy_export_CB : export scRNA-seq dataset to Cell Browser         #
#                                                                               #
#################################################################################


def scanpy_export_CB(adata, cb_outdir, name, clusterField = 'chosen_cluster', 
                     markerField = 'rank_genes_groups'):
  
  # Import modules
  print ("Importing modules ...\n")
  import sys
  import os
  import numpy as np
  import matplotlib
  import matplotlib.pyplot as plt
  import scanpy as sc
  import pandas as pd
  import re
  import cellbrowser.cellbrowser as cb
  from datetime import datetime
  
  
  # Data description
  print("Loading annData object: \n")
  print(adata)
  
  
  # Export data
  cb.scanpyToCellbrowser(adata, cb_outdir, name, clusterField, markerField)



#################################################################################
#                                                                               #
#                   8. scanpy_annotate : annotate cell clusters                 #
#                                                                               #
#################################################################################


def scanpy_annotate(adata, new_cluster_names, marker_genes, clusterId = "leiden", outDir = "./"):
  
  # Data
  print ("Basic information of the annData:")
  adata
  
  
  # Annotation
  adata.rename_categories(clusterId, new_cluster_names)
  print ("UMAP of cell types: ")
  sc.pl.umap(adata, color= clusterId, legend_loc= 'on data', title='', frameon=False, 
             save = outDir + "cell_types.png")
  print ("Dotplot of marker genes across cell types: ")
  sc.pl.dotplot(adata, marker_genes, groupby= clusterId, save = outDir + "dotplot_markers_celltypes.png")
  print ("Violin plot of marker genes across cell types: ")
  sc.pl.stacked_violin(adata, marker_genes, groupby= clusterId, 
                       rotation=90, save = outDir + "vlnplot_markers_celltypes.png")
  
  
  return adata



#################################################################################
#                                                                               #
#             9. scanpy_trajectory : trajectory analysis using Scanpy           #
#                                                                               #
#################################################################################


# Input :
# 1. denoising : whether denoising the data
# 2. key_genes : key genes to showcase
# 3. celltypes : cell type annotation labels


def scanpy_trajectory(adata, outDir = "./", denoising = False, key_genes = None, 
                      celltypes = None):
  
  # Import modules
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as pl
  from matplotlib import rcParams
  import scanpy as sc
  
  sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
  sc.logging.print_versions()
  sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), 
    facecolor='white')  # low dpi (dots per inch) yields small inline figures
  
  
  # Data
  adata.X = adata.X.astype('float64')  # this is not required and results will be comparable without it
  print ("annData: ")
  adata
  
  
  # Preprocessing and Visualization
  print ("Applying Zheng17 preprocessing pipeline ...")
  sc.pp.recipe_zheng17(adata)
  print ("Dimension reduction ...")
  sc.tl.pca(adata, svd_solver='arpack')
  print ("Finding neighbors and building graph ...")
  sc.pp.neighbors(adata, n_neighbors=4, n_pcs=20)
  sc.tl.draw_graph(adata)
  print ("Showing graph using force-directed layout: ")
  sc.pl.draw_graph(adata, color='paul15_clusters', legend_loc='on data', 
    save = outDir + "fa_graph.png")
  
  
  # Optional: Denoising the graph
  if denoising == True:
    sc.tl.diffmap(adata)
    sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')
    sc.tl.draw_graph(adata)
    print ("Showing graph after denoising using force-directed layout: ")
    sc.pl.draw_graph(adata, color='paul15_clusters', legend_loc='on data', 
      save = outDir + "fa_denoised_graph.png")
  
  
  # Clustering and PAGA
  print ("Clustering using Louvain ...")
  sc.tl.louvain(adata, resolution = 1.0)
  sc.tl.paga(adata, groups='louvain')
  print ("Showing PAGA graph of Louvain: ")
  sc.pl.paga(adata, color = "louvain", save = outDir + "clusters_paga.png")
  if key_genes != None:
    print ("Showing PAGA graph of key genes: ")
    sc.pl.paga(adata, color = key_genes, save = outDir + "keyGenes_paga.png")
  print ("Showing cluster labels: ")
  adata.obs['louvain'].cat.categories
  adata.obs['louvain_anno'] = adata.obs['louvain']
  print ("Annotating cell types ...")
  adata.obs['louvain_anno'].cat.categories = celltypes
  print ("Annotated cell types: ")
  adata.obs['louvain_anno'].cat.categories
  sc.tl.paga(adata, groups='louvain_anno')
  print ("PAGA of annotated cell types: ")
  sc.pl.paga(adata, threshold=0.03, show=False, save = outDir + "paga_celltyped.png")
  
  
  # Recomputing the embedding using PAGA-initialization
  sc.tl.draw_graph(adata, init_pos='paga')
  print ("PAGA graph of cell types:")
  sc.pl.draw_graph(adata, color= "louvain_anno", legend_loc='on data', 
    save = outDir + "paga_graph_celltypes.png")
  print ("PAGA graph of key genes: ")
  sc.pl.draw_graph(adata, color= key_genes, legend_loc='on data', 
    save = outDir + "paga_graph_keyGenes.png")
  
  
  return adata



#################################################################################
#                                                                               #
#                 10. scanpy_pt : pseudotime analysis using Scanpy              #
#                                                                               #
#################################################################################


# Input :
# 1. root_ct : cell type as root
# 2. gene_names : key genes to showcase
# 3. paths : cell paths in trajectory to showcase in trajectory


def scanpy_pt(adata, root_ct, gene_names, paths):
  
  # Import modules
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as pl
  from matplotlib import rcParams
  import scanpy as sc
  
  sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
  sc.logging.print_versions()
  sc.settings.set_figure_params(dpi=80, frameon=False, figsize=(3, 3), 
    facecolor='white')  # low dpi (dots per inch) yields small inline figures
  
  
  # Data
  print ("Basic information of annData: ")
  adata
  
  
  # Reconstructing gene changes along PAGA paths for a given set of genes
  print ("Selecting cells belonging to " + root_ct + " as root ...")
  adata.uns['iroot'] = np.flatnonzero(adata.obs['louvain_anno']  == root_ct)[0]
  sc.tl.dpt(adata)
  sc.pl.draw_graph(adata, color=['louvain_anno', 'dpt_pseudotime'], legend_loc='on data', 
    save = outDir + "pseudotime.png")
  adata.obs['distance'] = adata.obs['dpt_pseudotime']
  adata.obs['clusters'] = adata.obs['louvain_anno']  # just a cosmetic change
  adata.uns['clusters_colors'] = adata.uns['louvain_anno_colors']
  
  
  # Generate eheatmap
  print ("Generating heatmaps for three paths: ")
  _, axs = pl.subplots(ncols=3, figsize=(6, 2.5), gridspec_kw={'wspace': 0.05, 'left': 0.12})
  pl.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.2)
  for ipath, (descr, path) in enumerate(paths):
      _, data = sc.pl.paga_path(
          adata, path, gene_names,
          show_node_names=False,
          ax=axs[ipath],
          ytick_fontsize=12,
          left_margin=0.15,
          n_avg=50,
          annotations=['distance'],
          show_yticks=True if ipath==0 else False,
          show_colorbar=False,
          color_map='Greys',
          groups_key='clusters',
          color_maps_annotations={'distance': 'viridis'},
          title='{} path'.format(descr),
          return_data=True,
          show=False)
      data.to_csv(outDir + 'paga_path_{}.csv'.format(descr))
  pl.savefig(outDir + 'heatmap.png')
  pl.show()
  
  
  return adata



#################################################################################
#                                                                               #
#       11. pyscenic_grn : infer gene regulatory networks using pySCENIC        #
#                                                                               #
#################################################################################


def pyscenic_grn(DATA_FOLDER, RESOURCES_FOLDER, DATABASE_FOLDER, SCHEDULER, 
        DATABASES_GLOB, MOTIF_ANNOTATIONS_FNAME, MM_TFS_FNAME, SC_EXP_FNAME, 
        REGULONS_FNAME, MOTIFS_FNAME):
    
    # Import modules
    import os
    import glob # The glob module finds all the pathnames matching a specified pattern 
    # according to the rules used by the Unix shell
    
    import pickle # the process of converting a Python object into a byte stream to store it in a file/database
    import pandas as pd # Python Data Analysis Library
    import numpy as np # NumPy is a library for the Python programming language
    
    from dask.diagnostics import ProgressBar # An interactive dashboard containing 
    # many plots and tables with live information
    
    from arboreto.utils import load_tf_names
    from arboreto.algo import grnboost2
    
    from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
    from pyscenic.utils import modules_from_adjacencies, load_motifs
    from pyscenic.prune import prune2df, df2regulons
    from pyscenic.aucell import aucell
    
    import seaborn as sns # Seaborn is a library that uses Matplotlib underneath to plot graphs
    
    
    # Load expression matrix and databases
    print ('Loading expression matrix and databases ...')
    ex_matrix = pd.read_csv(SC_EXP_FNAME, sep='\t', header=0, index_col=0).T
    print ('Size of the expression matrix: ')
    ex_matrix.shape
    print ('Loading TF names ...')
    tf_names = load_tf_names(MM_TFS_FNAME)
    print ('Loading TF databases ...')
    db_fnames = glob.glob(DATABASES_GLOB)
    def name(fname):
        return os.path.splitext(os.path.basename(fname))[0]
    dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]
    dbs
    
    
    # Build GRN and regulons
    print ('Building adjacency matrix ...')
    adjacencies = grnboost2(ex_matrix, tf_names=tf_names, verbose=True)
    print ('Predicting co-expression modules ...')
    modules = list(modules_from_adjacencies(adjacencies, ex_matrix))
    
    
    # Calculate a list of enriched motifs and the corresponding target genes for all modules.
    print ('Adding TFs to gene co-expression modules ...')
    with ProgressBar():
        df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME)
    
    # Create regulons from this table of enriched motifs.
    regulons = df2regulons(df)
    
    # Save the enriched motifs and the discovered regulons to disk.
    df.to_csv(MOTIFS_FNAME)
    with open(REGULONS_FNAME, "wb") as f:
        pickle.dump(regulons, f)
    
    # The clusters can be leveraged via the dask framework:
    df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME, client_or_address=SCHEDULER)
    
    
    #  Enrichment of a regulon is measured as the Area Under the recovery Curve (AUC)
    print ('Measuring the enrichment of regulons via Area Under the recovery Curve (AUC) ...')
    auc_mtx = aucell(ex_matrix, regulons, num_workers=4)
    sns.clustermap(auc_mtx, figsize=(8,8))
