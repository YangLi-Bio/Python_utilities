#################################################################################
#                                                                               #
# Functions to analyze scMulti-omics data with the following characteristics    #
#                                                                               #
#################################################################################


# Parameters
script_dir = "/fs/ess/PCON0022/liyang/Python_utilities/Functions"



# Functions
# 1. vipcca_integrate_rna_atac : integrate RNA+ATAC using VIPCCA
# 2. glue_preprocess : GLUE processing for single-cell RNA + ATAC
# 3. glue_training : training model using GLUE
# 4. glue_reg_infer : gene regulatory inference using GLUE



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
  
  
  return adata_integrate



#################################################################################
#                                                                               #
#        2. glue_preprocess : GLUE processing for single-cell RNA+ATAC          #
#                                                                               #
#################################################################################


def glue_preprocess(rna, atac, gtf, outDir = "./"):
  
  # Modules
  import anndata as ad
  import networkx as nx
  import scanpy as sc
  import scglue
  from matplotlib import rcParams
  scglue.plot.set_publication_params()
  rcParams["figure.figsize"] = (4, 4)
  
  
  # Data
  print ("scRNA-seq data:")
  rna
  print ("scATAC-seq: ")
  atac
  
  
  # Preprocess scRNA-seq data
  print ("Preprocessing scRNA-seq data ...")
  rna.layers["counts"] = rna.X.copy()
  sc.pp.highly_variable_genes(rna, n_top_genes=2000, flavor="seurat_v3")
  sc.pp.normalize_total(rna)
  sc.pp.log1p(rna)
  sc.pp.scale(rna)
  sc.tl.pca(rna, n_comps=100, svd_solver="auto")
  sc.pp.neighbors(rna, metric="cosine")
  sc.tl.umap(rna)
  sc.pl.umap(rna, color = "cell_type", save = outDir + "RNA_celltypes.png")
  
  
  # Preprocess scATAC-seq data
  print ("Preprocessing scATAC-seq data ...")
  scglue.data.lsi(atac, n_components=100, n_iter=15)
  sc.pp.neighbors(atac, use_rep="X_lsi", metric="cosine")
  sc.tl.umap(atac)
  sc.pl.umap(atac, color="cell_type", save = outDir + "ATAC_celltypes.png")
  
  
  # Construct prior regulatory graph
  rna.var.head()
  scglue.data.get_gene_annotation(
      rna, gtf = gtf,
      gtf_by ="gene_name"
  )
  rna.var.loc[:, ["chrom", "chromStart", "chromEnd"]].head()
  atac.var_names[:5]
  split = atac.var_names.str.split(r"[:-]")
  atac.var["chrom"] = split.map(lambda x: x[0])
  atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
  atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)
  atac.var.head()
  
  
  # Graph construction
  print ("Consrtruct graphs ...")
  guidance = scglue.genomics.rna_anchored_guidance_graph(rna, atac)
  guidance
  scglue.graph.check_graph(guidance, [rna, atac])
  atac.var.head()
  
  
  return rna, atac, guidance



#################################################################################
#                                                                               #
#                 3. glue_training : training model using GLUE                  #
#                                                                               #
#################################################################################


def glue_training(rna, atac, guidance, outDir = "./"):
  
  # Modules
  from itertools import chain
  import anndata as ad
  import itertools
  import networkx as nx
  import pandas as pd
  import scanpy as sc
  import scglue
  import seaborn as sns
  from matplotlib import rcParams
  scglue.plot.set_publication_params()
  rcParams["figure.figsize"] = (4, 4)
  
  
  # Data
  print ("Basic information of scRNA-seq:")
  rna
  print ("Basic information of scATAC-seq:")
  atac
  print ("Guidance graph:")
  guidance
  
  
  # Configure data
  print ("Configuring the scRNA-seq data ...")
  scglue.models.configure_dataset(
      rna, "NB", use_highly_variable=True,
      use_layer="counts", use_rep="X_pca"
  )
  print ("Configuring the scATAC-seq data ...")
  scglue.models.configure_dataset(
      atac, "NB", use_highly_variable=True,
      use_rep="X_lsi"
  )
  print ("Building a subgraph composed of highly variable features ...")
  guidance_hvf = guidance.subgraph(chain(
      rna.var.query("highly_variable").index,
      atac.var.query("highly_variable").index
  )).copy()
  
  
  # Train GLUE model
  print ("Training GLUE model ...")
  glue = scglue.models.fit_SCGLUE(
      {"rna": rna, "atac": atac}, guidance_hvf,
      fit_kws={"directory": "glue"}
  )
  glue.save(outDir + "glue.dill")
  
  
  # Check integration diagnostics
  print ("Checking integration diagnostics ...")
  dx = scglue.models.integration_consistency(
      glue, {"rna": rna, "atac": atac}, guidance_hvf
  )
  dx
  _ = sns.lineplot(x="n_meta", y="consistency", 
    data = dx).axhline(y = 0.05, c = "darkred", ls = "--")
  
  
  # Apply model for cell and feature embedding
  print ("Applying model for cell and feature embedding ...")
  rna.obsm["X_glue"] = glue.encode_data("rna", rna)
  atac.obsm["X_glue"] = glue.encode_data("atac", atac)
  combined = ad.concat([rna, atac])
  sc.pp.neighbors(combined, use_rep="X_glue", metric="cosine")
  sc.tl.umap(combined)
  sc.pl.umap(combined, color=["cell_type", "domain"], wspace=0.65, save = outDir + "UMAP_integrated.png")
  feature_embeddings = glue.encode_graph(guidance_hvf)
  feature_embeddings = pd.DataFrame(feature_embeddings, index=glue.vertices)
  feature_embeddings.iloc[:5, :5]
  rna.varm["X_glue"] = feature_embeddings.reindex(rna.var_names).to_numpy()
  atac.varm["X_glue"] = feature_embeddings.reindex(atac.var_names).to_numpy()
  
  
  return rna, atac, guidance_hvf



#################################################################################
#                                                                               #
#           4. glue_reg_infer : gene regulatory inference using GLUE            #
#                                                                               #
#################################################################################


# Prepare Genome track ini file
def prepare_genome_track(outDir, link_file, bed_file, gtf_file):

  # Write ini file
  lines = [
    "[Score]",
    "file = " + outDir + link_file,
    "title = Score",
    "height = 2",
    "color = YlGnBu",
    "compact_arcs_level = 2",
    "use_middle = True",
    "file_type = links",
    "",
    "[ATAC]",
    "file = " + outDir + bed_file,
    "title = ATAC",
    "display = collapsed",
    "border_color = none",
    "labels = False",
    "file_type = bed",
    "",
    "[Genes]",
    "file = " + gtf_file,
    "title = Genes",
    "prefered_name = gene_name", 
    "height = 4",
    "merge_transcripts = True",
    "labels = True", 
    "max_labels = 100",
    "all_labels_inside = True",
    "style = UCSC",
    "file_type = gtf",
    "",
    "[x-axis]",
    "fontsize = 12"
  ]
  
  
  # Write file
  print ("Writing tracks to file: " + outDir + "tracks.ini ...")
  with open(outDir + 'tracks.ini', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')



def glue_reg_infer(rna, atac, gtf_file, guidance_hvf, key_gene, outDir = "./"):
  
  # Modules
  # import os
  import anndata as ad
  import networkx as nx
  import numpy as np
  import pandas as pd
  import scglue
  import seaborn as sns
  from IPython import display
  from matplotlib import rcParams
  from networkx.algorithms.bipartite import biadjacency_matrix
  from networkx.drawing.nx_agraph import graphviz_layout
  scglue.plot.set_publication_params()
  rcParams['figure.figsize'] = (4, 4)
  
  
  # Read intermediate results
  print ("Extracting the list of highly-variable features for future convenience ...")
  rna.var["name"] = rna.var_names
  atac.var["name"] = atac.var_names
  genes = rna.var.query("highly_variable").index
  peaks = atac.var.query("highly_variable").index
  
  
  # Cis-regulatory inference with GLUE feature embeddings
  print ("Cis-regulatory inference with GLUE feature embeddings ...")
  features = pd.Index(np.concatenate([rna.var_names, atac.var_names]))
  feature_embeddings = np.concatenate([rna.varm["X_glue"], atac.varm["X_glue"]])
  skeleton = guidance_hvf.edge_subgraph(
      e for e, attr in dict(guidance_hvf.edges).items()
      if attr["type"] == "fwd"
  ).copy()
  reginf = scglue.genomics.regulatory_inference(
      features, feature_embeddings,
      skeleton=skeleton, random_state=0
  )
  gene2peak = reginf.edge_subgraph(
      e for e, attr in dict(reginf.edges).items()
      if attr["qval"] < 0.05
  )
  
  
  # Visualize the inferred cis-regulatory regions
  print ("Visualize the inferred cis-regulatory regions ...")
  scglue.genomics.Bed(atac.var).write_bed(outDir + "peaks.bed", ncols=3)
  scglue.genomics.write_links(
      gene2peak,
      scglue.genomics.Bed(rna.var).strand_specific_start_site(),
      scglue.genomics.Bed(atac.var),
      outDir + "gene2peak.links", keep_attrs = ["score"]
  )
  prepare_genome_track(outDir, "gene2peak.links", "peaks.bed", gtf_file)
  # os.chdir(outDir)
  loc = rna.var.loc[key_gene]
  chrom = loc["chrom"]
  chromLen = loc["chromEnd"] - loc["chromStart"]
  chromStart = loc["chromStart"] - chromLen
  chromEnd = loc["chromEnd"] + chromLen
  !pyGenomeTracks --tracks ${outDir}/tracks.ini \
      --region {chrom}:{chromStart}-{chromEnd} \
      --outFileName ${outDir}/tracks.png 2> /dev/null
  print ("Displaying the genome tracks image ...")
  display.Image(outDir + "tracks.png")
  
  
  return gene2peak
