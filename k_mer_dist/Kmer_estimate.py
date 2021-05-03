import os
import sys
sys.path.insert(1, '/Users/weiwei/PycharmProjects/pythonProject/CmashKmerAbundance/KMC')
sys.path.insert(1, '/Users/weiwei/PycharmProjects/pythonProject/CmashKmerAbundance/KMC/bin')
import py_kmc_api as kmc
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np
from CMash import MinHash as MH
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import time
import statsmodels.datasets

def get_MH_data(n, k, genome_file, rev_comp=False):
    '''

    :param n:
    :param k: kmer size
    :param genome_file: fasta format
    :param rev_comp:
    :return:
    '''
    estimator = MH.CountEstimator(n=n, ksize=k, save_kmers='n', input_file_name=genome_file, rev_comp=rev_comp)
    counts = estimator._counts
    count_dict = dict()
    for count in counts:
        if count > 0:
            if count in count_dict.keys():
                count_dict[count]+=1
            else:
                count_dict[count] = 1
    normed_dict = dict()
    total_count = sum(count_dict.values())
    for k, v in count_dict.items():
        normed_dict[k] = count_dict[k] / total_count
    #print("minhash results:")
    #print(normed_dict)
    #print(len(normed_dict.keys()))
    #print("checking if MH estimate is correct:")
    #print(sum(count_dict.values()))
    #print(count_dict)
    return normed_dict

def get_kmc_data(k, genome_file, out_file, out_dir, verbose=False):
    #run from CmashKmerAbundance dir
    kmc_database = kmc.KMCFile()
    abs_path = os.getcwd()
    out_path = abs_path+'/'+out_dir+'/'+out_file
    os.system('kmc -k%d -cs1000 -b -v -fm -ci2 %s %s %s' %(k, genome_file, out_path , '.'))
    kmc_database.OpenForListing(out_path)
    kmer_obj = kmc.KmerAPI(kmc_database.Info().kmer_length)
    counter = kmc.Count()
    counter_dict = dict()
    while kmc_database.ReadNextKmer(kmer_obj, counter):
        if int(counter.value) in counter_dict.keys():
            counter_dict[int(counter.value)]+=1
        else:
            counter_dict[int(counter.value)] = 1
    print(len(counter_dict.keys()))
    print(counter_dict)
    normed_dict = dict()
    total_count = sum(counter_dict.values())
    for k, v in counter_dict.items():
        normed_dict[k] = counter_dict[k]/total_count
    #df = pd.DataFrame(list(normed_dict.items()), columns=['kmer_count', 'percentage'])
    #print(sum(normed_dict.values()))
    #print(df)
    #sns.histplot(x='kmer_count', y='percentage', binwidth=1, data=df)
    #plt.savefig('test1.png')
    print(normed_dict)
    return normed_dict

def get_distance(dict1, dict2, type='L1'):
    dict1_copy = deepcopy(dict1)
    dict2_copy = deepcopy(dict2)
    print(sum(dict2_copy.values()))
    diff1 = dict1_copy.keys() - dict2_copy.keys()
    if len(diff1) > 0:
        for k in diff1:
            dict2_copy[k] = 0
    diff2 = dict2_copy.keys() - dict1_copy.keys()
    if len(diff2) > 0:
        for k in diff2:
            dict1_copy[k] = 0
    dist = 0
    #print(dict1)
    #print(dict2)
    if type == "L1":
        for key in dict1_copy.keys():
           dist += np.abs(dict1_copy[key] - dict2_copy[key])
    elif type == "L2":
        for key in dict1_copy.keys():
            dist += (dict1_copy[key] - dict2_copy[key])**2
    elif type == "ratio":
        #get rid of 0
        for key in dict1_copy.keys():
            dict1_copy[key]+=1
            dict2_copy[key]+=1
        for key in dict1_copy.keys():
            dist+=np.abs(dict1_copy[key]-dict2_copy[key])/dict1_copy[key]
    elif type == "wasserstein":
        #convert to lists
        lst1 = np.zeros(max(dict1_copy.keys()))
        lst2 = np.zeros(max(dict2_copy.keys())) #actually same
        for k in dict1_copy.keys():
            lst1[k-1] = dict1_copy[k]
            lst2[k-1] = dict2_copy[k]
        dist = wasserstein_distance(lst1, lst2)
    else:
        print("distance type unrecognizable.")
    print(dist)
    return dist

def get_count_dict(count_list):
    count_dict = dict()
    for count in count_list:
        if count > 1:
            if count in count_dict.keys():
                count_dict[count] += 1
            else:
                count_dict[count] = 1
    normed_dict = dict()
    total_count = sum(count_dict.values())
    for k, v in count_dict.items():
        normed_dict[k] = count_dict[k] / total_count
    return normed_dict

def quick_dump(k_list, n, input_file):
    for k in k_list:
        pickle_file = 'k'+str(k)+'n'+str(n)+input_file+'.pickle'
        print(pickle_file)
        estimator = MH.CountEstimator(n=n, ksize=k, save_kmers='n', input_file_name=input_file, rev_comp=False)
        counts = estimator._counts
        with open(pickle_file, 'wb') as pf:
            pickle.dump(counts, pf)

def get_n_vs_dist_dataframe(k, input_file, n):
    df = pd.DataFrame(columns=['L1', 'L2', ], index=n)
    kmc_normed_dict = get_kmc_data(k, input_file, input_file+'_out', 'out')
    pickle_name = 'k'+str(k)+'n' + str(max(n))+input_file+'.pickle'
    with open(pickle_name, 'rb') as pf:
        count_list = pickle.load(pf)
    for n_value in n:
        estimated_normed_dict = get_count_dict(count_list[0:n_value])
        L1 = get_distance(kmc_normed_dict, estimated_normed_dict, 'L1')
        L2 = get_distance(kmc_normed_dict, estimated_normed_dict, 'L2')
        df.at[n_value, 'L1'] = L1
        df.at[n_value, 'L2'] = L2
    print(df)
    outfile = "n_vs_dict"+"k"+str(k)+".txt"
    df.to_csv(outfile, sep='\t', index=True)
    return

def how_is_k_doing(n, kmin, kmax, kstep, input_file, outfile):
    k_list = list(range(kmin,kmax,kstep))
    df_mh = pd.DataFrame(columns=['k', 'L1', 'L2', 'ratio', 'wasserstein', 'method'], index=k_list)
    df_ntcard = pd.DataFrame(columns=['k', 'L1', 'L2', 'ratio', 'wasserstein', 'method'], index=k_list)
    df_mh['method'] = "Cmash"
    df_ntcard['method'] = "ntCard"
    df_mh['k'] = k_list
    df_ntcard['k'] = k_list
    for k in k_list:
        pickle_name='k' + str(k) + 'n10000' + input_file + '.pickle'
        os.system('ntcard -k%d -p %s %s' %(k, 'srrntcard', input_file))
        time.sleep(10)
        with open(pickle_name, 'rb') as pf:
            count_list = pickle.load(pf)
        estimated_normed_dict = get_count_dict(count_list)
        kmc_normed_dict = get_kmc_data(k, input_file, input_file + '_out', 'out')
        mhL1 = get_distance(kmc_normed_dict, estimated_normed_dict, 'L1')
        mhL2 = get_distance(kmc_normed_dict, estimated_normed_dict, 'L2')
        mhratio = get_distance(kmc_normed_dict, estimated_normed_dict, 'ratio')
        mhwasser = get_distance(kmc_normed_dict, estimated_normed_dict, 'wasserstein')
        df_mh.at[k, 'L1'] = mhL1
        df_mh.at[k, 'L2'] = mhL2
        df_mh.at[k, 'ratio'] = mhratio
        df_mh.at[k, 'wasserstein'] = mhwasser
        nt_out = 'srrntcard' + '_k' + str(k) + '.hist'
        nt_dist = parse_ntcard_output(nt_out)
        ntL1 = get_distance(kmc_normed_dict, nt_dist, 'L1')
        ntL2 = get_distance(kmc_normed_dict, nt_dist, 'L2')
        ntratio = get_distance(kmc_normed_dict, nt_dist, 'ratio')
        ntwasser = get_distance(kmc_normed_dict, nt_dist, 'wasserstein')
        df_ntcard.at[k, 'L1'] = ntL1
        df_ntcard.at[k, 'L2'] = ntL2
        df_ntcard.at[k, 'ratio'] = ntratio
        df_ntcard.at[k, 'wasserstein'] = ntwasser
    df = pd.concat([df_mh, df_ntcard])
    df.to_csv(outfile, sep='\t', index=True)
    print(df)

def kmc_cmash_compare(k, n, input_file):
    kmc_normed_dict = get_kmc_data(k, input_file, input_file+'_out', 'out')
    #minhash estimate
    estimator = MH.CountEstimator(n=n, ksize=k, save_kmers='n', input_file_name=input_file)
    real_dist = pd.DataFrame(list(kmc_normed_dict.items()), columns=['kmer_count', 'percentage'])
    sns.barplot(x='kmer_count', y='percentage', data=real_dist)
    plt.savefig('quicklook_real.png')
    counts = estimator._counts
    estimated_normed_dict = get_count_dict(counts)
    #quick look at distribution
    df = pd.DataFrame(list(estimated_normed_dict.items()), columns=['kmer_count', 'percentage'])
    sns.barplot(x='kmer_count', y='percentage', data=df)
    plt.savefig('quicklook.png')
    #####
    print(sum(estimated_normed_dict.values()))
    print(counts)
    print(get_distance(kmc_normed_dict, estimated_normed_dict, 'wasserstein'))


def quicker_dump(input_file):
    n=10000
    for k in [25, 50,75]:
        pickle_file = 'k' + str(k) + 'n10000' + input_file + '.pickle'
        print(pickle_file)
        estimator = MH.CountEstimator(n=n, ksize=k, save_kmers='n', input_file_name=input_file, rev_comp=False)
        counts = estimator._counts
        with open(pickle_file, 'wb') as pf:
            pickle.dump(counts, pf)

def parse_ntcard_output(file):
    nt_dict = dict()
    with open(file, 'r') as f:
        f.readline()
        f.readline()
        f.readline()
        for line in f.readlines():
            line = line.strip()
            dist = line.split('\t')
            nt_dict[int(dist[0])] = int(dist[1])
        normed_dict = dict()
        total_count = sum(nt_dict.values())
        for k, v in nt_dict.items():
            normed_dict[k] = nt_dict[k] / total_count
        return normed_dict


def test_bb_batch():
    n_range = [100, 500, 1000, 10000]
    k25df = pd.DataFrame(columns=['n', 'L2', 'k'])
    k50df = pd.DataFrame(columns=['n', 'L2', 'k'])
    k75df = pd.DataFrame(columns=['n', 'L2', 'k'])
    n_col = []
    for n in n_range:
        n_col = n_col + [n]*10
    k25df['n'] = n_col
    k50df['n'] = n_col
    k75df['n'] = n_col
    k25df['k'] = 25
    k50df['k'] = 50
    k75df['k'] = 75
    k25df['L2'] = _get_L2_col(25, n_range)
    k50df['L2'] = _get_L2_col(50, n_range)
    k75df['L2'] = _get_L2_col(75, n_range)
    df_combined = pd.concat([k25df, k50df, k75df])
    df_combined.to_csv('df_for_boxplot.txt', sep="\t")

def _get_L2_col(k, n_range):
    L2_col = []
    for n in n_range:
        #10 repeats
        for i in range(10):
            bbfile = 'bbsim' + str(i+1) + '.fasta'
            kmc_normed_dict = get_kmc_data(k, bbfile, bbfile + '_out', 'out')
            pickle_file = 'k'+str(k) + 'n10000' + bbfile + '.pickle'
            with open(pickle_file, 'rb') as pf:
                count_list = pickle.load(pf)
            estimated_normed_dict = get_count_dict(count_list[0:n])
            L2 = get_distance(kmc_normed_dict, estimated_normed_dict, 'L2')
            print("n=%d k=%d L2=%f " %(n, k, L2))
            L2_col.append(L2)
    return L2_col

def get_boxplot_from_file(file, x, y, hue, outfile):
    df = pd.read_table(file)
    #sns.boxplot(x="n", y="L2", hue="k", data=df)
    #plt.savefig("L2_against_n.png")
    sns.lineplot(x=x, y=y, data=df, hue=hue)
    plt.savefig(outfile)

def test_lambda_estimate():
    file = "bbsim4.fasta"
    kmc_dict = get_kmc_data(25, file, file+"_out", "out")
    kmc_df = pd.DataFrame(list(kmc_dict.items()), columns=['kmer_count', 'percentage'])
    print(kmc_df.percentage.mean())
    lambda_est = 1./kmc_df.percentage.mean()
    print(lambda_est)
    pickle_file = "k25n10000bbsim4.fasta.pickle"
    with open(pickle_file, 'rb') as pf:
        count_list = pickle.load(pf)
    estimated_normed_dict = get_count_dict(count_list[0:10000])
    cmash_df = pd.DataFrame(list(estimated_normed_dict.items()), columns=['kmer_count', 'percentage'])
    lambda_est2 = 1./cmash_df.percentage.mean()
    print(lambda_est2)




if __name__ == "__main__":
    #kmc_result = get_kmc_data(60, 'SRR172902_1.fastq', 'SRR172902_out', 'out')
    #MH_result = get_MH_data(1000, 50, 'bbsim2.fasta', rev_comp=False)
    #print(MH_result)
    #get_distance(kmc_result, MH_result, 'L1')
    #get_distance(kmc_result, MH_result, 'L2')
    #quick_dump([70, 60, 48, 30, 21, 15], 100000, 'SRR172902_1.fastq') #not quick at all, takes forever
    #n = [500, 1000, 5000, 10000, 30000, 50000, 80000, 100000]
    #for k in [15, 21, 30, 48, 60, 70]:
    #    get_n_vs_dist_dataframe(k, 'SRR172902_1.fastq', n)
    #quicker_dump('SRR172902_1.fastq')
    #kmc_cmash_compare(25, 1000, 'bbsim2.fasta')
    #for i in range(9):
    #    file = 'bbsim' + str(i+1) + '.fasta'
    #    quicker_dump(file)
    #test_bb_batch()
    #get_boxplot_from_file('df_for_boxplot.txt')
    #how_is_k_doing(1000, 7, 75, 5, 'SRR172902_1.fastq', 'howsk.txt')
    #get_boxplot_from_file('howsk.txt', 'k', 'wasserstein', 'method', "k_agains_wasserstein.png")
    test_lambda_estimate()