#################################################################################
#                                                                               #
# Functions to analyze scMulti-omics data with the following characteristics    #
#                                                                               #
#################################################################################


# Parameters
script_dir = "/fs/ess/PCON0022/liyang/Python_utilities/Functions"



# Functions
# 1. vipcca_integrate_rna_atac : integrate RNA + ATAC using VIPCCA
# 2. glue_preprocess : GLUE processing for single-cell RNA + ATAC
# 3. glue_training : training model using GLUE
# 4. glue_reg_infer : gene regulatory inference using GLUE
# 5. multiVI_integrate_impute : integrate RNA + ATAC and impute missing modality
# 6. muon_tri_integrate : integrate RNA + ATAC + protein from TEA-seq using muon



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



#################################################################################
#                                                                               #
#       5. multiVI_integrate_impute : integrate RNA + ATAC and impute missing   #
#          modality                                                             #
#                                                                               #
#################################################################################


def multiVI_integrate_impute(adata_rna, atac_atac, atac_paired, genesToImpute, outDir = './'):
    
    # Modules
    import scvi
    import numpy as np
    import scanpy as sc
    
    scvi.settings.seed = 420
    
    %config InlineBackend.print_figure_kwargs={'facecolor' : "w"}
    %config InlineBackend.figure_format='retina'
    
    
    # Data
    print ('RNA modality: ')
    adata_rna
    print ('ATAC modality: ')
    adata_atac
    print ('Paired modality: ')
    adata_paired
    
    
    # We can now use the organizing method from scvi to concatenate these anndata
    print ('Concatenating the modalities ...')
    adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired, adata_rna, adata_atac)
    adata_mvi.obs
    
    
    # Sort features
    print ('Sorting the features so that genes occur before peaks ....')
    adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()
    adata_mvi.var
    
    
    # Filter features to remove those that appear in fewer than 1% of the cells
    print ('Data size before filtering: ')
    print(adata_mvi.shape)
    sc.pp.filter_genes(adata_mvi, min_cells=int(adata_mvi.shape[0] * 0.01))
    print ('Data size after filtering: ')
    print(adata_mvi.shape)
    
    
    # Setup and Training MultiVI
    print ('Setting up the annData ...')
    scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key='modality')
    mvi = scvi.model.MULTIVI(
        adata_mvi,
        n_genes=(adata_mvi.var['modality']=='Gene Expression').sum(),
        n_regions=(adata_mvi.var['modality']=='Peaks').sum(),
    )
    mvi.view_anndata_setup()
    mvi.train()
    
    
    # Save and Load MultiVI models
    print ('Saving the model to ' + outDir + 'trained_multivi')
    mvi.save(outDir + "trained_multivi")
    # mvi = scvi.model.MULTIVI.load("trained_multivi", adata=adata_mvi)
    
    
    # Extracting and visualizing the latent space
    print ('Visualizing the latent space ...')
    adata_mvi.obsm["MultiVI_latent"] = mvi.get_latent_representation()
    sc.pp.neighbors(adata_mvi, use_rep="MultiVI_latent")
    sc.tl.umap(adata_mvi, min_dist=0.2)
    sc.pl.umap(adata_mvi, color='modality', save = outDir + 'latent_space.png')
    
    
    # Impute missing modalities
    print ('Imputing missing modalities for ' + len(genesToImpute) + ' genes ...')
    imputed_expression = mvi.get_normalized_expression()
    for gene in genesToImpute:
        print ('Imputing gene ' + gene + " ...")
        gene_idx = np.where(adata_mvi.var.index == gene)[0]
        adata_mvi.obs[gene + '_imputed'] = imputed_expression.iloc[:, gene_idx]
        sc.pl.umap(adata_mvi, color= gene + '_imputed', save = outDir + gene + '_imputed_umap.png')
    
    
    return adata_mvi, imputed_expression



#################################################################################
#                                                                               #
# 6. muon_tri_integrate : integrate RNA + ATAC + protein from TEA-seq using     #
#    muon                                                                       #
#                                                                               #
#################################################################################


# Process protein expression
def muon_process_protein(mdata, celltype1, celltype2, outDir = './'):
    
    # Modules
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import muon as mu
    import muon.atac as ac
    import muon.prot as pt
    import matplotlib
    from matplotlib import pyplot as plt
    plt.rcParams['figure.dpi'] = 100
    
    
    # Protein modality (epitopes)
    print ("Protein modality processing for protein ...")
    prot = mdata["prot"]
    print ('Preserving original counts in a layer before the normalisation for protein ...')
    prot.layers['counts'] = prot.X
    print ('Specifying features corresponding to the isotype controls for protein ...')
    isotypes = prot.var_names.values[["Isotype" in v for v in prot.var_names]]
    print(isotypes)
    print ('Normalizing counts for protein ...')
    pt.pp.dsb(mdata, prot_raw, isotype_controls=isotypes, random_state=1)
    print ('Plotting values to visualise the effect of normalisation for protein ...')
    sc.pl.scatter(mdata['prot'], x="prot:" + celltype1, y="prot:" + celltype2, layers='counts', 
            save = outDir + 'scatter_prot_' + celltype1 + '_' + celltype2 + '_raw.png')
    sc.pl.scatter(mdata['prot'], x="prot:" + celltype1, y="prot:" + celltype2, 
            save = outDir + 'scatter_prot_' + celltype1 + '_' + celltype2 + '_normalized.png')
    
    
    # Downstream analysis
    print ('Downstream analysis for protein ...')
    sc.tl.pca(prot)
    sc.pl.pca(prot, color=['prot:' + celltype1, 'prot:' + celltype2], 
            save = outDir + 'pca_prot_' + celltype1 + '_' + celltype2 + '.png')
    sc.pp.neighbors(prot)
    sc.tl.umap(prot, random_state=1)
    sc.pl.umap(prot, color=['prot:' + celltype1, 'prot:' + celltype2], 
            save = outDir + 'umap_prot_' + celltype1 + '_' + celltype2 + '.png')
    
    
    return prot
    


# Quality control of RNA
def muon_quality_ctr_rna(mdata, outDir = './'):
    
    # Modules
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import muon as mu
    import muon.atac as ac
    import muon.prot as pt
    import matplotlib
    from matplotlib import pyplot as plt
    plt.rcParams['figure.dpi'] = 100
    
    
    print ("Protein modality processing for RNA ...")
    rna = mdata.mod['rna']
    rna
    print ('Quality control for RNA ...')
    rna.var['mt'] = rna.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    mu.pl.histogram(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
            save = outDir + 'qc_rna.png')
    
    
    return rna



# Process RNA expression
def muon_process_rna(mdata, genes, qc_ar = [200, 2500, 500, 5000, 30], outDir = './'):
    
    # Modules
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import muon as mu
    import muon.atac as ac
    import muon.prot as pt
    import matplotlib
    from matplotlib import pyplot as plt
    plt.rcParams['figure.dpi'] = 100
    
    
    # QC
    mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 10)  # gene detected at least in 10 cells
    mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= qc_ar[0]) & (x < qc_ar[1]))
    mu.pp.filter_obs(rna, 'total_counts', lambda x: (x > qc_ar[2]) & (x < qc_ar[3]))
    mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < qc_ar[4])
    mu.pl.histogram(rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
        save = outDir + 'qc_rna_after.png')
    
    
    # Normalisation
    print ('Normalizing RNA counts ...')
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    
    
    # Feature selection
    print ('Feature selection ...')
    sc.pp.highly_variable_genes(rna, min_mean=0.02, max_mean=4, min_disp=0.5)
    sc.pl.highly_variable_genes(rna, save = outDir + 'rna_highly_variable_genes.png')
    
    
    # Scaling
    print ('Scaling the RNA counts ...')
    rna.layers["lognorm"] = rna.X.copy()
    sc.pp.scale(rna, max_value=10)
    
    
    # Downstream analysis
    print ('Downstream analysis ...')
    sc.tl.pca(rna, svd_solver='arpack')
    sc.pl.pca(rna, color= genes, layer="lognorm", save = outDir + 'rna_pca_genes.png')
    sc.pl.pca_variance_ratio(rna, log=True, save = outDir + 'rna_pca_var.png')
    sc.pp.neighbors(rna, n_neighbors=10, n_pcs=20)
    sc.tl.leiden(rna, resolution=.75)
    sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)
    sc.pl.umap(rna, color="leiden", legend_loc="on data", save = outDir + 'umap_rna_clusters.png')
    
    
    return rna



# Process ATAC data
def muon_process_atac(mdata, genes, frag_file, meta_file, qc_ar = [200, 2500, 500, 5000, 30], outDir = './'):
    
    # Modules
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import muon as mu
    import muon.atac as ac
    import muon.prot as pt
    import matplotlib
    from matplotlib import pyplot as plt
    plt.rcParams['figure.dpi'] = 100
    
    
    # Load data
    atac = mdata['atac']
    atac
    print ('Loading ATAC fragments ...')
    ac.tl.locate_fragments(atac, frag_file)
    print ('Adding metadata ...')
    metadata = pd.read_csv(meta_file)
    pd.set_option('display.max_columns', 500)
    metadata.head()
    atac.obs = atac.obs.join(metadata.set_index("original_barcodes"))
    
    
    # Quality control
    print ('Quality control for ATAC ...')
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
    mu.pl.histogram(atac, ['total_counts', 'n_genes_by_counts'], save = outDir + 'atac_qc.png')
    mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= 5)  # a peak is detected in 5 cells or more
    mu.pp.filter_obs(atac, 'total_counts', lambda x: (x >= qc_ar[0]) & (x <= qc_ar[1]))
    mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= qc_ar[2]) & (x <= qc_ar[3]))  # number of peaks per cell
    mu.pl.histogram(atac, ['total_counts', 'n_genes_by_counts'], save = outDir + 'atac_qc_after.png')
    ac.tl.nucleosome_signal(atac, n=1e6, barcodes="barcodes")
    print ('Calculating chromosome signal ...')
    mu.pl.histogram(atac, "nucleosome_signal", kde=False, save = outDir + 'nucleosome_signal.png')
    print ('Calculating TSS enrichment ...')
    tss = ac.tl.tss_enrichment(mdata, n_tss=1000, barcodes="barcodes")  # by default, 
    # features=ac.tl.get_gene_annotation_from_rna(mdata)
    
    ac.pl.tss_enrichment(tss, save = outDir + 'atac_tss.png')
    
    # To avoid issues with nan being converted to 'nan' when the column is categorical,
    # we explicitely convert it to str
    mu.pp.filter_obs(atac, atac.obs.barcodes.astype(str) != 'nan')
    
    
    # Normalisation
    print ('Normalizing the ATAC data ...')
    atac.layers["counts"] = atac.X
    sc.pp.normalize_per_cell(atac, counts_per_cell_after=1e4)
    sc.pp.log1p(atac)
    
    
    # Feature selection
    print ('Selecting features from ATAC ...')
    sc.pp.highly_variable_genes(atac, min_mean=0.05, max_mean=1.5, min_disp=.5)
    sc.pl.highly_variable_genes(atac, save = outDir + 'atac_highly_variable_features.png')
    
    
    # Scaling
    print ('Scaling the ATAC data ...')
    atac.layers["lognorm"] = atac.X.copy()
    
    
    # Downstream analysis
    sc.pp.scale(atac)
    sc.tl.pca(atac)
    sc.pl.pca(atac, color=["n_genes_by_counts", "n_counts"], layer="lognorm", 
            save = outDir + 'atac_pca.png')
    sc.pp.neighbors(atac, n_neighbors=10, n_pcs=30)
    sc.tl.leiden(atac, resolution=.5)
    sc.tl.umap(atac, spread=1.5, min_dist=.5, random_state=30)
    sc.pl.umap(atac, color=["leiden", "n_genes_by_counts"], legend_loc="on data", 
            save = outDir + 'atac_umap_celltypes.png')
    
    
    return atac



# Multi-omics analyses
def muon_tri_integrate(mdata, genes, outDir = "./"):
    
    # Modules
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import muon as mu
    import muon.atac as ac
    import muon.prot as pt
    import matplotlib
    from matplotlib import pyplot as plt
    plt.rcParams['figure.dpi'] = 100
    
    
    # Data
    print ('Basic information of trimodality data: ')
    mdata
    mu.pp.intersect_obs(mdata)
    
    
    # Multiplex clustering
    print ('Perform multiplex clustering ...')
    mdata.uns = dict()
    mu.tl.leiden(mdata, resolution=[1., 1., 1.], random_state=1, key_added="leiden_multiplex")
    mu.pl.embedding(mdata, basis="rna:X_umap", color=["prot:" + genes[0], "prot:" + genes[1], "leiden_multiplex", 
                    "rna:leiden", "atac:leiden"], 
            save = outDir + 'umap_multimodal_clusters.png')
    
    
    # Multi-omics factor analysis
    print ('Performing multi-omics factor analysis ...')
    prot.var["highly_variable"] = True
    mdata.update()
    mu.tl.mofa(mdata, outfile= outDir + "models/pbmc_w3_teaseq.hdf5")
    mu.pl.mofa(mdata, color=['prot:' + genes[0], 'prot:' + genes[1]], save = outDir + 'prot_mofa.png')
    sc.pp.neighbors(mdata, use_rep="X_mofa", key_added='mofa')
    sc.tl.umap(mdata, min_dist=.2, random_state=1, neighbors_key='mofa', save = outDir + 'mofa_embeddings.png')
    
    
    # Interpreting the model
    print ('Interpreting the MOFA+ model ...')
    mu.pl.mofa_loadings(mdata, save = outDir + 'top_features_mofa.png')
    
    
    # Weighted nearest neighbours
    print ('Conducting weighted nearest neighbours ...')
    for m in mdata.mod.keys():
        sc.pp.neighbors(mdata[m])
    mu.pp.neighbors(mdata, key_added='wnn')
    sc.tl.leiden(mdata, resolution=.55, neighbors_key='wnn', key_added='leiden_wnn')
    mu.tl.umap(mdata, random_state=10, neighbors_key='wnn')
    mu.pl.umap(mdata, color="leiden_wnn", save = outDir + 'wnn_umap_leiden.png')
    
    
    return mdata



# Annotate cell types using muon
def muon_annot_celltypes(mdata, genes, new_cluster_names):
    
    # Variables
    # 1. genes : marker genes of various cell types
    # 2. new_cluster_names : dictory of assignment from cell clusters to cell types
    
    
    # Modules
    import os
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import muon as mu
    import muon.atac as ac
    import muon.prot as pt
    import matplotlib
    from matplotlib import pyplot as plt
    plt.rcParams['figure.dpi'] = 100
    
    
    # Feature plots
    print ('Generating feature plots of marker genes ...')
    mu.pl.embedding(mdata, basis="X_wnn_umap", color=list(map(
        lambda x: "prot:" + x, genes
    )), save = outDir + 'feature_plots_markers.png')
    
    
    # Cell type annotation
    print ('Annotating cell types ...')
    mdata.obs['celltype'] = mdata.obs.leiden_wnn.astype("str").values
    mdata.obs.celltype = mdata.obs.celltype.astype("category")
    mdata.obs.celltype = mdata.obs.celltype.cat.rename_categories(new_cluster_names)
    mdata.obs.celltype.cat.reorder_categories([set( val for dic in new_cluster_names for val in dic.values())], 
            inplace=True)
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(mdata.obs.celltype.cat.categories)))
    mdata.uns["celltype_colors"] = list(map(matplotlib.colors.to_hex, colors))
    mu.pl.umap(mdata, color="celltype", frameon=False, title="UMAP(WNN)", save = outDir + 'umap_celltypes.png')
    
    
    return mdata
