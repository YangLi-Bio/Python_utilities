#################################################################################
#                                                                               #
# Functions to analyze scMulti-omics data with the following characteristics    #
#                                                                               #
#################################################################################


# Parameters
script_dir = "/fs/ess/PCON0022/liyang/Python_utilities/Functions"



# Functions
# 1. vipcca_integrate_rna_atac : integrate RNA+ATAC using VIPCCA



#################################################################################
#                                                                               #
#       1. vipcca_integrate_rna_atac : integrate RNA+ATAC using VIPCCA          #
#                                                                               #
#################################################################################


def vipcca_integrate_rna_atac(adata_rna, adata_atac, res_path. outDir = "./"):
  
  # Modules
  import vipcca.model.vipcca as vip
  import vipcca.tools.utils as tl
  import vipcca.tools.plotting as pl
  import vipcca.tools.transferLabel as tfl
  import scanpy as sc
  from sklearn.preprocessing import LabelEncoder
  import os
  
  import matplotlib
  matplotlib.use('TkAgg')
  
  # Command for Jupyter notebooks only
  %matplotlib inline
  import warnings
  warnings.filterwarnings('ignore')
  from matplotlib.axes._axes import _log as matplotlib_axes_logger
  matplotlib_axes_logger.setLevel('ERROR')
  
  
  # Data introduction
  print ("Basic information of RNA:")
  adata_rna
  print ("Basic information of ATAC:")
  adata_atac
  
  
  # Data preprocessing
  print ("Processing data ...")
  adata_all= tl.preprocessing([adata_rna, adata_atac])
  
  
  # VIPCCA integration
  print ("Integrating RNA and ATAC ...")
  handle = vip.VIPCCA(adata_all=adata_all,
                           res_path= res_path,
                           mode='CVAE',
                           split_by="_batch",
                           epochs=20,
                           lambda_regulizer=2,
                           batch_input_size=64,
                           batch_input_size2=14,
                           )
  adata_integrate=handle.fit_integrate()
  
  
  # Cell type prediction
  print ("Cell type annotation ...")
  atac=tfl.findNeighbors(adata_integrate)
  adata_atac.obs['celltype'] = atac['celltype']
  adata = adata_rna.concatenate(adata_atac)
  adata_integrate.obs['celltype'] = adata.obs['celltype']
  
  
  # UMAP Visualization
  sc.pp.neighbors(adata_integrate, use_rep='X_vipcca')
  sc.tl.umap(adata_integrate)
  sc.set_figure_params(figsize=[5.5, 4.5])
  sc.pl.umap(adata_integrate, color=['_batch', 'celltype'], save = outDir + "batch_celltypes.png")
  
  
  adata_integrate
