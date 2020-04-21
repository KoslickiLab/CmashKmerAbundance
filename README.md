# CmashKmerAbundance
A repository demonstrating how CMash (containment MinHash) can be used to produce k-mer abundance histograms

Note: Some files are used with absolute path, if necessary please change them by searching "/gpfs/scratch/xbz5174/short_term_work_Feb/"

## Scripts:
k_mer_dist/k_mer_dist_test.py: estimate k-mer abundance distributions with CMash
CMash/MinHash_by_seq.py: increase sketch counts by sequences of k-mers instead of hash values (mainly in line 155-158)
k_mer_dist/K_mer_dist_test_by_seq.py: estimate k-mer abundance distributions with CMash (which counts by k-mer sequences)
k_mer_dist/k_mer_dist_run.py: run and debug the two k_mer_dist_test*.py scripts.
k_mer_dist/hist_plot_for_SOTA.py (deprecated): It is used for reading ntCard and KmerEst outputs and make histograms and distance matrices for them. It's deprecated. Please use k_mer_dist_run.py.

## Data and results (not included):
k_mer_dist/kmc_global_count: k-mer abundance distributions computed by KMC and CMash
k_mer_dist/kmc_global_count/MH_counts: MinHash sketches
k_mer_dist/sota_res: k-mer abundances computed by ntCard and KmerEst
data_real: not provided, the data used are SRR072232 and SRR172902