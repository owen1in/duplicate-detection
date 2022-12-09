#%% Import packages and data preparation
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
import numpy as np
import random
import json
import math
import time
import re

printt = False
z = open("TVs-all-merged.json")
z_data = json.load(z)
z.close()

tvs = []
data = []
models = []
models_store = []
for tv in z_data:
    tvs.append(tv)
    model = []
    for i in range(len(z_data[tv])):
        model_site = []
        model_site.append(z_data[tv][i]["shop"][0])
        model_site.append(z_data[tv][i]["title"])
        model_site.append(
            " ".join(list(z_data[tv][i]["featuresMap"].values()))
        )
        model.append(model_site)
        models.append(tv)
        models_store.append(str(tv + "_{}".format(z_data[tv][i]["shop"][0])))
    data.append(model)
no_of_dups = 0
for x in range(len(models)):
    for y in range(x + 1, len(models)):
        if models[x] == models[y]:
            no_of_dups += 1
no_of_products = len(models)
max_comparisons = round((len(models) * (len(models) - 1)) / 2)
#%% Define functions
def hash_f(a, b, key, p):
    """
    Args:
        a (int): Random integer
        b (int): Random integer
        key (int): Hash key, row number of binary matrix
        p (int): prime number (p > len(binary matrix))

    Returns:
        int: Hash value (bucket number)

    """
    return (a + b * key) % p


def is_prime(n):
    """
    Args:
        n (int): Integer to check wheter it is a prime

    Returns:
        bool: True or False

    """
    for i in range(2, n):
        if (n % i) == 0:
            return False
    return True


def threshold(r, b):
    """
    Args:
        r (int): Number of rows
        b (int): Number of bands

    Returns:
        float: Threshold

    """
    return (1 / b) ** (1 / r)


def jaccard_dist(cp1, cp2):
    """
    Args:
        cp1 (array): Candidate pair 1
        cp2 (array): Candidate pair 1

    Returns:
        float: Jaccard dissimilarity between the candidate pairs

    """
    intersection = np.logical_and(cp1, cp2)
    union = np.logical_or(cp1, cp2)
    if not float(union.sum()) == 0:
        similarity = intersection.sum() / float(union.sum())
    else:
        similarity = -999998
    return 1 - similarity


#%% Data cleaning
# Unit variantions to "inch", "hz" and "lb" as well as punctuations
z_inches = ['"', "”", "inches", "-inch", ' "', " inches", " inch", "'"]
z_hertz = ["hertz", " hertz", " hz", "-hz"]
z_pounds = [
    "pounds",
    "lbs.",
    "-lbs",
    "-lb",
    " lb",
    " lbs",
    " lbs.",
    " pounds",
]
z_punctuations = ["(", ")", ",", ";", "°", "[", "]"]
z_replaced = [z_inches, z_hertz, z_punctuations, z_pounds]  #!!!
z_replacer = ["inch", "hz", "", "lb"]  #!!!

for tv in data:
    for i in range(len(tv)):  # ith site on which the tv is sold
        site = tv[i]  # list 0: store {a,n,b,t}, 1: title, 2: features
        # Cleaning titles and features
        for section in range(
            1, 3
        ):  # section is either the title or the features
            site[section] = site[section].lower().replace("/", " ")
            for j in range(len(z_replaced)):
                replaced = z_replaced[j]
                replacer = z_replacer[j]
                for replace_value in replaced:
                    if replace_value in site[section]:
                        site[section] = site[section].replace(
                            replace_value, replacer
                        )
#%% Find Model Words from title and features
titles = [title[1] for site in data for title in site]
model_words_title = [
    title_part
    for title in titles
    for title_part in title.split()
    if re.search(
        "([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)",
        title_part,
    )
]
z_first_title_word = [first.split()[0].strip() for first in titles]
z_brands = [
    "samsung",
    "lg",
    "sony",
    "vizio",
    "panasonic",
    "tcl",
    "philips",
    "hisense",
    "toshiba",
    "sansui",
    "coby",
    "avue",
    "insignia",
    "magnavox",
    "jvc",
    "optoma",
    "sharp",
    "rca",
    "dynex",
    "mitsubishi",
]
model_words = model_words_title.copy()
model_words.extend(z_brands)
model_words = list(set(model_words))

#%% Bootstrap
def bootstrap(data):
    """
    Args:
        data (list): Data on which bootstrap sample will be made

    Returns:
        train_data (list): Train section of data
        test_data (list): Test section of data

    """
    data_single = [j for i in data for j in i]
    bootstrap = [
        random.randrange(1, len(data_single), 1)
        for i in range(len(data_single))
    ]
    train_idx = list(set(bootstrap))
    test_idx = list(set([i for i in range(len(data_single))]) - set(train_idx))
    test_data = [data_single[i] for i in test_idx]
    train_data = [data_single[i] for i in train_idx]
    tvs_train = [models[i] for i in train_idx]
    tvs_test = [models[i] for i in test_idx]
    max_comparisons_train = round(len(train_idx) * (len(train_idx) - 1) / 2)
    max_comparisons_test = round(len(test_idx) * (len(test_idx) - 1) / 2)
    return (
        train_data,
        test_data,
        tvs_train,
        tvs_test,
        max_comparisons_train,
        max_comparisons_test,
    )


#%% Create Binary Vector Representation
def bin_vec(data, tvs):
    """
    Args:
        data (list): Input data
        tvs (list): All models in the data

    Returns:
        bin_df (dataframe): Binary matrix
        no_of_dups (int): True number of duplicates in the data set
        model_store (list): List of models with store indicator

    """
    vectors = []
    stores = []
    for i in range(len(tvs)):
        model = tvs[i]
        tv = data[i]
        vector = [str(model + "_{}".format(tv[0]))]
        for mw in model_words:
            if mw in tv[1]:
                vector.append(1)
            else:
                vector.append(0)
        vectors.append(vector)
        stores.append(tv[0])
        # models.append(model)
    bin_df = pd.DataFrame(vectors).T
    model_store = bin_df.iloc[0]
    bin_df = bin_df[1:]
    bin_df.columns = model_store
    bin_df["mw"] = model_words
    bin_df = bin_df.set_index("mw")
    no_of_dups = 0
    for x in range(len(tvs)):
        for y in range(x + 1, len(tvs)):
            if tvs[x] == tvs[y]:
                no_of_dups += 1
    return bin_df, no_of_dups, model_store


#%% MinHash
def minhash(df):
    """
    Args:
        df (dataframe): Binary matrix

    Returns:
        M (dataframe): Signature matrix
        n (int): Length of signature matrix
        binary_matrix (dataframe): Cleaned binary matrix

    """
    binary_matrix = df.copy().reset_index(drop=True)
    mask = binary_matrix.sum(axis=1) > 0
    binary_matrix = binary_matrix[mask]
    binary_matrix.index += 1
    r = binary_matrix.shape[0]
    c = binary_matrix.shape[1]
    n = round(r * 0.4)  #!!!
    p = r + 1
    while is_prime(p) == False:
        p += 1
    constant = [random.randint(1, p) for a in range(n)]
    scalar = [random.randint(1, p) for b in range(n)]

    M = pd.DataFrame(np.zeros((n, c)) + 999999)

    for row in range(r):
        row_i = binary_matrix.iloc[row, :]
        hash_function_result = pd.DataFrame(
            [hash_f(constant[ab], scalar[ab], row + 1, p) for ab in range(n)]
        )
        mask = (row_i.eq(1)).reset_index(drop=True)
        indices = mask.index[mask == True].tolist()
        for idx in indices:
            replacer = pd.concat([M[idx], hash_function_result], axis=1).min(
                axis=1
            )
            M[idx] = replacer
    return M, n, binary_matrix


#%% LSH
def lsh(n, r, M):
    """
    Args:
        n (int): Length of signature matrix
        r (int): Number of rows per band
        M (dataframe): Signature matrix

    Returns:
        cps (list): List of candidate pairs

    """
    cps = []
    b = math.ceil(n / r)
    for band in range(b):
        cps_dict = {}
        if band == (b - 1):
            start, end = (band * r), n
        else:
            start, end = (band * r), (band * r + r)
        df_band = M.iloc[start:end, :]
        df_band = df_band.T[df_band.T.duplicated(keep=False)].T
        for index in df_band.columns:
            column = df_band.loc[:, index].astype(int).astype(str)
            column_string = column.str.cat(sep="")
            if column_string in cps_dict:
                cps_dict[column_string].append(index)
            else:
                cps_dict[column_string] = [index]
        for bucket in cps_dict:
            candidates = cps_dict[bucket]
            pairs = list(
                map(
                    sorted,
                    [
                        [a, b]
                        for idx, a in enumerate(candidates)
                        for b in candidates[idx + 1 :]
                    ],
                )
            )
            cps.extend(pairs)
    cps = [list(x) for x in set(tuple(x) for x in cps)]
    return cps


#%% LSH Evaluation
def f1_star(cps, models, no_of_dups):
    """
    Args:
        cps (list): List of candidate pairs
        models (list): List of tv models
        no_of_dups (int): Number of true duplicates in the data set

    Returns:
        p_q (float): Pair quality
        p_c (float): Pair completeness
        f1_star (float): F1* score
        no_of_cps (int): Number of candidate pairs (number of comparisons)

    """
    no_of_cps = len(cps)
    found_dups = len(
        [1 for combi in cps if models[combi[0]] == models[combi[1]]]
    )
    p_q = found_dups / no_of_cps
    p_c = found_dups / no_of_dups
    f1_star = (2 * p_q * p_c) / (p_q + p_c)
    return p_q, p_c, f1_star, no_of_cps


#%% Dissimilarity Matrix
def dissim(cps, tvs, binary_matrix):
    """
    Args:
        cps (list): List of candidate pairs
        tvs (list): List of models
        binary_matrix (dataframe): Binary matrix representation of data set

    Returns:
        dissim (dataframe): Dissimilarity matrix

    """
    no_of_products = len(tvs)
    dissim = pd.DataFrame(
        data=999999, index=range(no_of_products), columns=range(no_of_products)
    )
    binary_matrix = binary_matrix.astype(int).reset_index(drop=True)
    binary_matrix.columns = range(no_of_products)
    for cp in cps:
        candidate1 = binary_matrix.iloc[:, cp[0]]
        candidate2 = binary_matrix.iloc[:, cp[1]]
        distance = jaccard_dist(candidate1, candidate2) * 100
        dissim.iat[cp[0], cp[1]] = distance
        dissim.iat[cp[1], cp[0]] = distance
    dissim = dissim.values
    np.fill_diagonal(dissim, 0)
    dissim = pd.DataFrame(dissim)
    dissim.columns = tvs
    dissim.index = tvs
    return dissim


#%% Clustering
def cluster(dt, dissim, tvs):
    """
    Args:
        dt (float): Distance threshold for agglomerative clustering
        dissim (dataframe): Dissimilarity matrix
        tvs (list): List of tv models

    Returns:
        cluster_dict (dictionary): Dictionary of clusters

    """

    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        linkage="complete",
        distance_threshold=dt * 100,
    ).fit(dissim.values)
    no_of_clusters = max(clustering.labels_) + 1
    clusters = pd.DataFrame(clustering.labels_, columns=["cluster"])
    clusters.index = tvs

    cluster_dict = {}
    for cluster in range(no_of_clusters):
        cluster_dict[cluster] = list(
            clusters[clusters.cluster == cluster].index
        )
    return cluster_dict


#%% Cluster Evaluation
def f1(cluster_dict, no_of_dups):
    """
    Args:
        cluster_dict (dictionary): Dictionary of clusters
        no_of_dups (int): True number of duplicates in the data set

    Returns:
        precision (float): Precision score
        recall (float): Recall score
        f1 (float): F1 score
        no_comparisons (int): Number of comparisons made

    """
    no_comparisons = 0
    dups_found = 0
    for i in cluster_dict:
        all_combis = [
            [a, b]
            for idx, a in enumerate(cluster_dict[i])
            for b in cluster_dict[i][idx + 1 :]
        ]
        no_comparisons += len(all_combis)
        for possible_pair in all_combis:
            if possible_pair[0] == possible_pair[1]:
                dups_found += 1
    precision = dups_found / no_comparisons
    recall = dups_found / no_of_dups
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, no_comparisons


#%% Optimization
# z_tick_gs = time.time()
# grid = [
#     [0.2, 5],
#     [0.2, 6],
#     [0.2, 7],
#     [0.2, 8],
#     [0.2, 9],
#     [0.2, 10],
#     [0.25, 5],
#     [0.25, 6],
#     [0.25, 7],
#     [0.25, 8],
#     [0.25, 9],
#     [0.25, 10],
#     [0.3, 5],
#     [0.3, 6],
#     [0.3, 7],
#     [0.3, 8],
#     [0.3, 9],
#     [0.3, 10],
#     [0.35, 5],
#     [0.35, 6],
#     [0.35, 7],
#     [0.35, 8],
#     [0.35, 9],
#     [0.35, 10],
#     [0.4, 5],
#     [0.4, 6],
#     [0.4, 7],
#     [0.4, 8],
#     [0.4, 9],
#     [0.4, 10],
#     [0.45, 5],
#     [0.45, 6],
#     [0.45, 7],
#     [0.45, 8],
#     [0.45, 9],
#     [0.45, 10],
#     [0.5, 5],
#     [0.5, 6],
#     [0.5, 7],
#     [0.5, 8],
#     [0.5, 9],
#     [0.5, 10],
# ]  # [dt, r]
# thresholds = [(1 / (len(model_words) / i)) ** (1 / i) for i in range(6, 16)]
# grid_bar = []
# no_of_bs = 5
# for grid_i in grid:
#     five_bootstrap_fones = []
#     for bstrap in range(no_of_bs):
#         print(
#             "--------\n\n Gridpoint {}/{}: {}\n\n Bootstrap {}/{}".format(
#                 grid.index(grid_i) + 1, len(grid), grid_i, bstrap + 1, no_of_bs
#             )
#         )
#         r = grid_i[1]
#         dt = grid_i[0]
#         (
#             train_data,
#             test_data,
#             tvs_train,
#             tvs_test,
#             max_compares_train,
#             max_compares_test,
#         ) = bootstrap(data)
#         df_train, no_of_dups_train, tvs_store_train = bin_vec(
#             train_data, tvs_train
#         )
#         M_train, n_train, binmatrix_train = minhash(df_train)
#         cps_train = lsh(n_train, r, M_train)  # rrr
#         dissim_train = dissim(cps_train, tvs_store_train, binmatrix_train)
#         pq_train, pc_train, f1star_train, comparisons_made_star = f1_star(
#             cps_train, tvs_train, no_of_dups_train
#         )
#         print(
#             "pq = {}, pc = {}, f1* = {}".format(
#                 round(pq_train, 2), round(pc_train, 2), round(f1star_train, 4)
#             )
#         )
#         clusters_train = cluster(dt, dissim_train, tvs_train)  # dtdtdt
#         precision_train, recall_train, f1_train, comparisons_made = f1(
#             clusters_train, no_of_dups_train
#         )
#         print(
#             "pr = {}, re = {}, f1 = {}".format(
#                 round(precision_train, 2),
#                 round(recall_train, 2),
#                 round(f1_train, 4),
#             )
#         )
#         five_bootstrap_fones.append(f1_train)
#     f_bar = mean(five_bootstrap_fones)
#     grid_bar.append(f_bar)
# opt_paras = grid[grid_bar.index(max(grid_bar))]
# z_tock_gs = time.time()
# z_elapsed_bv = z_tock_gs - z_tick_gs
# print("Gridsearch performed in {} minutes".format(round(z_elapsed_bv / 60)))

#%% Final Run
# data_single = [j for i in data for j in i]
# df, no_of_dups, tvs_store = bin_vec(data_single, models)
# M, n, binmatrix = minhash(df)
# # binmatrix = binmatrix.sample(frac=1).reset_index(drop=True) #!!!
# r = 5  # optimal number of rows 5 or 6

# cps = lsh(n, r, M)  # rrr
# pq, pc, f1star, comparisons_made_star = f1_star(cps, models, no_of_dups)
# dissim_df = dissim(cps, models_store, binmatrix)

# dt = 0.2  # optimal distance_threshold 0.2 0r 0.25
# clusters = cluster(dt, dissim_df, models)  # dtdtdt
# precision, recall, f1_score, comparisons_made = f1(clusters, no_of_dups)
# print(
#     "pq = {}, pc = {}, f1* = {}, fraction = {}".format(
#         round(pq, 2),
#         round(pc, 2),
#         round(f1star, 4),
#         round(comparisons_made_star / max_comparisons, 4),
#     )
# )
# print(
#     "pr = {}, re = {}, f1 = {}, fraction = {}".format(
#         round(precision, 2),
#         round(recall, 2),
#         round(f1_score, 4),
#         round(comparisons_made / max_comparisons, 4),
#     )
# )
#%% Final
z_tick_final = time.time()
rs = [r for r in range(1, 10)]
rs.extend([r for r in range(10, 23, 3)])
rs.extend([25, 50, 250, 350])
# rs.sort(reverse=True)
no_of_bs = 10
dt = 0.2
fractions = []
f1s = []
f1_stars = []
pqs = []
pcs = []
precisions = []
recalls = []
for r in rs:
    bootstrap_fractions = []
    bootstrap_fones = []
    bootstrap_fone_stars = []
    bootstrap_pqs = []
    bootstrap_pcs = []
    bootstrap_precisions = []
    bootstrap_recalls = []
    for bstrap in range(no_of_bs):
        print(
            "--------\n\n r {}/{}: {}\n\n Bootstrap {}/{}".format(
                rs.index(r) + 1, len(rs), r, bstrap + 1, no_of_bs
            )
        )
        (
            train_data,
            test_data,
            tvs_train,
            tvs_test,
            max_compares_train,
            max_compares_test,
        ) = bootstrap(data)
        df_train, no_of_dups_train, tvs_store_train = bin_vec(
            train_data, tvs_train
        )
        M_train, n_train, binmatrix_train = minhash(df_train)
        cps_train = lsh(n_train, r, M_train)  # rrr
        dissim_train = dissim(cps_train, tvs_store_train, binmatrix_train)
        pq_train, pc_train, f1star_train, comparisons_made_star = f1_star(
            cps_train, tvs_train, no_of_dups_train
        )
        fraction_lsh = round(comparisons_made_star / max_compares_train, 8)
        print(
            "pq = {}, pc = {}, f1* = {}, fraction = {}".format(
                round(pq_train, 2),
                round(pc_train, 2),
                round(f1star_train, 4),
                fraction_lsh,
            )
        )
        clusters_train = cluster(dt, dissim_train, tvs_train)  # dtdtdt
        precision_train, recall_train, f1_train, comparisons_made = f1(
            clusters_train, no_of_dups_train
        )
        print(
            "pr = {}, re = {}, f1 = {}".format(
                round(precision_train, 2),
                round(recall_train, 2),
                round(f1_train, 4),
            )
        )
        bootstrap_fractions.append(round(fraction_lsh, 4))
        bootstrap_fones.append(round(f1_train, 4))
        bootstrap_fone_stars.append(round(f1star_train, 4))
        bootstrap_pqs.append(round(pq_train, 2))
        bootstrap_pcs.append(round(pc_train, 2))
        bootstrap_precisions.append(round(precision_train, 2))
        bootstrap_recalls.append(round(recall_train, 2))
    # frac_bar = mean(bootstrap_fractions)
    # f_bar = mean(bootstrap_fones)
    # f_star_bar = mean(bootstrap_fone_stars)
    # pq_bar = mean(bootstrap_pqs)
    # pc_bar = mean(bootstrap_pcs)
    # precision_bar = mean(bootstrap_precisions)
    # recall_bar = mean(bootstrap_recalls)

    # fractions.append(frac_bar)
    # f1s.append(f_bar)
    # f1_stars.append(f_star_bar)
    # pqs.append(pq_bar)
    # pcs.append(pc_bar)
    # precisions.append(precision_bar)
    # recalls.append(recall_bar)
    fractions.append(bootstrap_fractions)
    f1s.append(bootstrap_fones)
    f1_stars.append(bootstrap_fone_stars)
    pqs.append(bootstrap_pqs)
    pcs.append(bootstrap_pcs)
    precisions.append(bootstrap_precisions)
    recalls.append(bootstrap_recalls)

z_tock_final = time.time()
z_elapsed_final = z_tock_final - z_tick_final
print("Final performed in {} minutes".format(round(z_elapsed_final / 60)))
