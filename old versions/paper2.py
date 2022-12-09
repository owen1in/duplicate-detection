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
            site[section] = site[section].lower()
            site[section] = site[section].replace("/", " ")
            site[section] = site[section].replace("-", " ")
            for j in range(len(z_replaced)):
                replaced = z_replaced[j]
                replacer = z_replacer[j]
                for replace_value in replaced:
                    if replace_value in site[section]:
                        site[section] = site[section].replace(
                            replace_value, replacer
                        )

#%% Find Model Words from title and features
# model_words = []
# for tv in data:
#     for i in range(len(tv)):  # ith site on which the tv is sold
#         site = tv[i]  # list 0: store {a,n,b}, 1: title, 2: features
#         title_split = site[1].split()
#         for title_part in title_split:
#             if re.search(
#                 "([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)",
#                 title_part,
#             ):
#                 model_words.append(title_part)
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
z_new_header = df.iloc[0]
df = df[1:]
df.columns = z_new_header
df["mw"] = model_words
df = df.set_index("mw")
a = df.T

z_tock = time.time()
z_elapsed = z_tock - z_tick
print("Binary vectors obtained in {} seconds".format(round(z_elapsed, 2)))


b = z_fours[0]
A = df[
    [
        b + "_a",
        b + "_b",
        b + "_n",
    ]
]
B = A[A.sum(axis=1) != 0]

#%% MinHash
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

constant = [a for a in range(1, 102)]
scalar = [b for b in range(102, 202)]
p = r
while is_prime(p) == False:
    p += 1

M = pd.DataFrame(np.zeros((2, c)) + 99999999)
for row in range(r):
    hash_function_results = pd.DataFrame()
    row_i = binary_matrix.iloc[row, :]
    min_replace = pd.DataFrame()
    for ab in range(2):  # 100 is just random test
        hash_function_results = pd.concat(
            [
                hash_function_results,
                pd.DataFrame(
                    [
                        hash_f(constant[ab], scalar[ab], i, p)
                        for i in range(1, c + 1)
                    ]
                ),
            ],
            axis=1,
        )
        min_replace = pd.concat(
            [
                min_replace,
                pd.DataFrame(
                    hash_function_results.iloc[:, ab].values * row_i.values
                ),
            ],
            axis=1,
        )
    min_replacer = pd.DataFrame(min_replace.min(axis=1))
    M_i = pd.DataFrame(M.iloc[row, :])
    M_i.loc[(M_i > min_replacer) & (min_replacer != 0)]
