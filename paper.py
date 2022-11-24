#%% Import packages and data preparation
import pandas as pd
import json
from collections import Counter
import re
import statistics
from sklearn.cluster import KMeans, DBSCAN
from datasketch import MinHash
import time
import random
import numpy as np

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
model_words_feature = []
# features_split = site[2].split()
# z_inch_extracted = [
#     inch for inch in features_split if (re.findall("inch$", inch))
# ]
# z_hz_extracted = [
#     hz for hz in features_split if (re.findall("hz$", hz))
# ]
# z_lb_extracted = [
#     lb for lb in features_split if (re.findall("lb$", lb))
# ]
# model_words.extend(z_inch_extracted)
# model_words.extend(z_hz_extracted)
# model_words.extend(z_lb_extracted)
model_words = model_words_title.copy()
model_words.extend(model_words_feature)
model_words = list(set(model_words))

#%% Create Binary Vector Representation
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

# b = z_fours[0]
# A = df[
#     [
#         b + "_a",
#         b + "_b",
#         b + "_n",
#     ]
# ]
# B = A[A.sum(axis=1) != 0]

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
k = round(r / 2)

p = r
while is_prime(p) == False:
    p += 1

constant = [random.randint(1, p) for a in range(k)]
scalar = [random.randint(1, p) for b in range(k)]

M = pd.DataFrame(np.zeros((k, c)) + 99999999)

for row in range(r):
    row_i = binary_matrix.iloc[row, :]
    hash_function_result = pd.DataFrame(
        [hash_f(constant[ab], scalar[ab], row + 1, p) for ab in range(k)]
    )
    mask = (row_i.eq(1)).reset_index(drop=True)
    indices = mask.index[mask == True].tolist()
    for idx in indices:
        replacer = pd.concat([M[idx], hash_function_result], axis=1).min(
            axis=1
        )
        M[idx] = replacer
M.columns = z_model_store

z_tock = time.time()
z_elapsed = z_tock - z_tick
print("Signature matrix obtained in {} seconds".format(round(z_elapsed, 2)))
