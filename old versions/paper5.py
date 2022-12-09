#%% Import packages and data preparation
from sklearn.metrics import accuracy_score as asc
from sklearn.metrics import jaccard_score as jsc
from sklearn.metrics import f1_score as fsc

# from collections import defaultdict
from statistics import mean
import pandas as pd
import numpy as np
import random
import json
import math
import time
import re


z_tick = time.time()
z = open("TVs-all-merged.json")
z_data = json.load(z)
z.close()

z_fours = ["47LW5600", "55LW5600", "65LW6500", "LEDTV2326"]

tvs = []
data = []
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
    data.append(model)
#%% Data cleaning
# Unit variantions to "inch", "hz" and "lb" as well as punctuations
z_inches = ['"', "inches", "-inch", ' "', " inches", " inch", "'"]
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
z_punctuations = [",", ";", "(", ")", "°"]
z_replaced = [z_inches, z_pounds, z_hertz, z_punctuations]
z_replacer = ["inch", "lb", "hz", ""]

for tv in data:
    for i in range(len(tv)):  # ith site on which the tv is sold
        site = tv[i]  # list 0: strore {a,n,b}, 1: title, 2: features
        # cleaning titles and features
        for section in range(
            1, 3
        ):  # section is either the title or the features
            site[section] = (
                site[section].lower().replace("/", " ").replace("-", " ")
            )
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
features = [feature[2] for site in data for feature in site]
feature_words = [
    feature_part
    for feature in features
    for feature_part in feature.split()
    if re.search("(^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$)", feature_part)
]
z_inch_extracted = [
    inch for inch in feature_words if (re.search("inch$", inch))
]
z_hz_extracted = [hz for hz in feature_words if (re.search("hz$", hz))]
z_lb_extracted = [lb for lb in feature_words if (re.search("lb$", lb))]
z_first_title_word = [first.split()[0].strip() for first in titles]
z_first_title_word = list(set(z_first_title_word))
model_words_feature = []
model_words_feature.extend(z_inch_extracted)
model_words_feature.extend(z_hz_extracted)
model_words_feature.extend(z_lb_extracted)

model_words = model_words_title.copy()
model_words.extend(model_words_feature)
model_words.extend(z_first_title_word)
model_words = list(set(model_words))

#%% Create Binary Vector Representation
models = []
vectors = []
for i in range(len(tvs)):
    model = tvs[i]
    tv = data[i]
    for i in range(len(tv)):  # ith site on which the tv is sold
        site = tv[i]  # list 0: store {a,n,b}, 1: title, 2: features
        vector = [str(model + "_{}".format(site[0]))]
        for mw in model_words:
            if (mw in site[1]) or (mw in site[2]):
                vector.append(1)
            else:
                vector.append(0)
        vectors.append(vector)
        models.append(model)

# quality_check = []
# for i in vectors:
#     quality_check.append(sum(i))
# statistics.mean(quality_check)
# statistics.quantiles(quality_check)

df = pd.DataFrame(vectors).T
z_model_store = df.iloc[0]
df = df[1:]
df.columns = z_model_store
df["mw"] = model_words
df = df.set_index("mw")
a = df.T

z_tock = time.time()
z_elapsed = z_tock - z_tick
print("Binary vectors obtained in {} seconds".format(round(z_elapsed, 2)))

#%% MinHash
z_tick = time.time()


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


binary_matrix = df.copy().reset_index(drop=True)
binary_matrix.index += 1
r = binary_matrix.shape[0]
c = binary_matrix.shape[1]
n = round(r * 0.75)  # no rows in sig matrix

p = r + 1
while is_prime(p) == False:
    # Find nearest prime where p > r
    p += 1

constant = [random.randint(1, p) for a in range(n)]
scalar = [random.randint(1, p) for b in range(n)]

M = pd.DataFrame(np.zeros((n, c)) + 99999999)

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
# M.columns = z_model_store # RENAME COLUMN HEADERS

z_tock = time.time()
z_elapsed = z_tock - z_tick
print("Signature matrix obtained in {} seconds".format(round(z_elapsed, 2)))

#%% LSH
def threshold(r, b):
    """
    Args:
        r (int): Number of rows
        b (int): Number of bands

    Returns:
        float: Threshold

    """
    return (1 / b) ** (1 / r)


cps = []
rows = []
bands = []
thresholds = []
r = 4
b = n / r
while threshold(r, b) < 0.9:
    rows.append(r)
    bands.append(b)
    thresholds.append(threshold(r, b))
    r += 2
    b = n / r
closest_to_50 = min(thresholds, key=lambda x: abs(x - 0.5))
if closest_to_50 > 0.5:
    idx_closest_to_50 = thresholds.index(closest_to_50) - 1
else:
    idx_closest_to_50 = thresholds.index(closest_to_50)
r = 8  # rows[idx_closest_to_50]
b = math.ceil(n / r)

test = []
z_tick_lsh = time.time()
for band in range(b):
    z_tick = time.time()
    cps_dict = {}
    # Define start and end indices of band
    if band == (b - 1):
        start, end = (band * r), n
    else:
        start, end = (band * r), (band * r + r)
    # Slice band
    df_band = M.iloc[start:end, :]
    df_band = df_band.T[df_band.T.duplicated(keep=False)].T
    # df_band.sort_values(list(df_band.index), inplace=True, axis=1)
    for index in df_band.columns:
        column = df_band.loc[:, index]
        column_string = round(column).apply(str).str.cat(sep="")
        if column_string in cps_dict:
            cps_dict[column_string].append(index)
        else:
            cps_dict[column_string] = [index]
    test.append(cps_dict)
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
    z_tock = time.time()
    z_elapsed = z_tock - z_tick
    print(
        "Candidates for band {} ({}%) obtained in {} seconds".format(
            band, round(band / b * 100), round(z_elapsed, 2)
        )
    )
cps = [list(x) for x in set(tuple(x) for x in cps)]
z_tock_lsh = time.time()
z_elapsed_lsh = z_tock_lsh - z_tick_lsh
print("---\nCandidates obtained in {} seconds".format(round(z_elapsed_lsh, 2)))
# 1361!!!

#%% LSH Evaluation
stripped_ascs = []
count = 0
denom = len(cps)
for cp in cps:
    mask = binary_matrix.iloc[:, cp].sum(axis=1) > 0
    stripped = binary_matrix.iloc[:, cp][mask]
    stripped_ascs.append(
        asc(stripped.iloc[:, 0].to_list(), stripped.iloc[:, 1].to_list())
    )
    count += 1
    print("{}%".format(round(100 * count / denom)))


accuracy = [
    asc(
        binary_matrix.iloc[:, cp[0]].to_list(),
        binary_matrix.iloc[:, cp[1]].to_list(),
    )
    for cp in cps
]
cps_accurate = pd.DataFrame(cps)
cps_accurate["accuracy"] = accuracy
cps_accurate = cps_accurate[cps_accurate["accuracy"] > 0.99]
count = 0
ascs_of_correct = []
test = pd.DataFrame()
for combi in range(len(cps_accurate)):
    idx1 = cps_accurate.iloc[combi, 0]
    idx2 = cps_accurate.iloc[combi, 1]
    model1 = models[idx1]
    model2 = models[idx2]
    print(
        "({}%) {}\n{}\n{}\n---".format(
            round(combi / len(cps_accurate) * 100),
            model1,
            model2,
            model1 == model2,
        )
    )
    if model1 == model2:
        count += 1
        ascs_of_correct.append(
            asc(
                binary_matrix.iloc[:, idx1].to_list(),
                binary_matrix.iloc[:, idx2].to_list(),
            )
        )
        testt = pd.DataFrame([idx1, idx2, model1, model2]).T
        test = pd.concat([test, testt])
test = test.reset_index(drop=True)
print("Real dups: {} out of {} candidates".format(count, len(cps_accurate)))

# words_matrix = binary_matrix.copy()
# words_matrix["mw"] = model_words
# words_matrix = words_matrix.set_index("mw")
# df_mw = pd.DataFrame(model_words)
# words_matrix = words_matrix.mul(df_mw.values[:, 0], axis=0)

count = 0
for i in z_data:
    if len(z_data[i]) == 2:
        count += 1
    if len(z_data[i]) == 3:
        count += 3
    if len(z_data[i]) == 4:
        count += 6
