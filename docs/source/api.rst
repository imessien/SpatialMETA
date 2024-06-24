API
===

Preprocessing
-------------

.. autofunction:: spatialmeta.pp.read_sm_csv_as_anndata
.. autofunction:: spatialmeta.pp.get_mz_reference
.. autofunction:: spatialmeta.pp.read_sm_imzml_as_anndata
.. autofunction:: spatialmeta.pp.merge_sm_pos_neg
.. autofunction:: spatialmeta.pp.calculate_qc_metrics_sm
.. autofunction:: spatialmeta.pp.filter_cells_sm
.. autofunction:: spatialmeta.pp.filter_metabolites_sm
.. autofunction:: spatialmeta.pp.new_spot_sample
.. autofunction:: spatialmeta.pp.spot_align_byknn
.. autofunction:: spatialmeta.pp.joint_adata_sm_st
.. autofunction:: spatialmeta.pp.normalize_total_joint_adata_sm_st
.. autofunction:: spatialmeta.pp.spatial_variable_joint_adata_sm_st
.. autofunction:: spatialmeta.pp.spatial_variable
.. autofunction:: spatialmeta.pp.rank_gene_and_metabolite_groups
.. autofunction:: spatialmeta.pp.corrcoef_stsm_inall
.. autofunction:: spatialmeta.pp.corrcoef_stsm_ingroup
.. autofunction:: spatialmeta.pp.spatial_distance_cluster
.. autofunction:: spatialmeta.pp.calculate_dot_df
.. autofunction:: spatialmeta.pp.metabolite_annotation
.. autofunction:: spatialmeta.pp.merge_and_assign_var_data
.. autofunction:: spatialmeta.pp.calculate_metabolite_enrichment
.. autofunction:: spatialmeta.pp.calculate_min_diam
.. autofunction:: spatialmeta.pp.add_obs_to_adata
.. autofunction:: spatialmeta.pp.add_hvf_to_jointadata
.. autofunction:: spatialmeta.pp.calculate_scale_factor
.. autofunction:: spatialmeta.pp.spot_transform_by_manual


Model
-----

Alignment Model for ST and SM
~~~~~~~~~~~~~~~~~~
This class is designed to align spatial transcriptomics (ST) and spatial metabolomics (SM) data. Firstly the model will learn a separate latent space for ST and SM. Then, the model will learn a linear transformation to align the two coordinate system.

.. autoclass:: spatialmeta.model.AlignmentVAE
.. autofunction:: spatialmeta.model.AlignmentVAE.fit_vae
.. autofunction:: spatialmeta.model.AlignmentVAE.fit_alignment


Integration Model for ST and SM
~~~~~~~~~~~~~~~~~~
This class is designed to handle spatial transcriptomics (ST) and spatial metabolomics (SM) data. The model learn a shared latent space to predict spatial sub-clusters characterized by unique transcriptional and metabolic states.

.. autofunction:: spatialmeta.model.ConditionalVAE
.. autofunction:: spatialmeta.model.ConditionalVAE.fit
.. autofunction:: spatialmeta.model.ConditionalVAE.get_latent_embedding
.. autofunction:: spatialmeta.model.ConditionalVAE.get_normalized_expression

Integration Model for SM Only
~~~~~~~~~~~~~~~~~~
This class is designed to handle spatial metabolomics (SM) data. The model learn a shared latent space to predict spatial sub-clusters characterized by unique metabolic states.

.. autofunction:: spatialmeta.model.ConditionalVAESM
.. autofunction:: spatialmeta.model.ConditionalVAESM.fit
.. autofunction:: spatialmeta.model.ConditionalVAESM.get_latent_embedding
.. autofunction:: spatialmeta.model.ConditionalVAESM.get_normalized_expression


Plotting
----------
.. autofunction:: spatialmeta.pl.make_colormap
.. autofunction:: spatialmeta.pl.create_fig
.. autofunction:: spatialmeta.pl.create_subplots
.. autofunction:: spatialmeta.pl.plot_spot_sm_st
.. autofunction:: spatialmeta.pl.plot_newdot_sm_st
.. autofunction:: spatialmeta.pl.plot_markerfeature
.. autofunction:: spatialmeta.pl.plot_marker_gene_metabolite
.. autofunction:: spatialmeta.pl.plot_corrcoef_stsm_inall
.. autofunction:: spatialmeta.pl.plot_corrcoef_stsm_ingroup
.. autofunction:: spatialmeta.pl.plot_spatial_deconvolution
.. autofunction:: spatialmeta.pl.plot_gene_corrcoef_sm_ingroup
.. autofunction:: spatialmeta.pl.plot_metabolite_corrcoef_st_ingroup
.. autofunction:: spatialmeta.pl.plot_volcano_corrcoef_gene
.. autofunction:: spatialmeta.pl.plot_volcano_corrcoef_metabolite
.. autofunction:: spatialmeta.pl.plot_trajectory_with_arrows
.. autofunction:: spatialmeta.pl.plot_clustermap_with_smoothing
.. autofunction:: spatialmeta.pl.plot_features_trajectory
.. autofunction:: spatialmeta.pl.plot_network
.. autofunction:: spatialmeta.pl.Wrapper
.. autofunction:: spatialmeta.pl.Wrapper.to_plotly

