
# Evaluation of feature selection and ranking methods' stability in the context of discovery proteomics data

Scripts for the Double Cross Validation described in my thesis.

The script was designed to handle datasets with small number of samples (~50) and bigger number of variables (~800). It uses many classifiers and Heavy feature selection procedures. Considering a laptop with i7: the ranking may take minutes/hours and the prioritization based on small sets of features may take days. In my study, 5 days.

# Would you like to contribute?

**Any performance and design improvements are welcome!** When my thesis is public available, I will post a link here.

![General Pipeline to score the proteins and further rank them using all the information.](images/general_pipeline.png)

![The RFA version developed for my thesis showed to be more Stable and rank better our true biomarkers than the classic RFE method.](images/dcv_rfe_vs_rfa.png)

![For each training data set (color), count the number of times that a Protein appeared as top-10 in the 40 ranks. The greater is the number of colors and size of bars, more stable is the protein.](images/dcv_number_of_times_in_top_10.png)

![Heatmap showing the highest 50 average scores of each protein for each Train dataset (column).](images/dcv_scores_highest_50_mean_heatmap.png)

![Top-10 proteins, selected from a rank of good and stable proteins.](images/heatmap_10best_svg.png)


