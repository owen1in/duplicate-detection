#%% Import packages and data preparation
import pandas as pd
import json
from collections import Counter
import re
import statistics
from sklearn.cluster import KMeans, DBSCAN
from datasketch import MinHash

z = open("TVs-all-merged.json")
data = json.load(z)
z.close()

z_fours = ["47LW5600", "55LW5600", "65LW6500", "LEDTV2326"]

tvs = []
datalist = []
for tv in data:
    tvs.append(tv)
    model = []
    for i in range(len(data[tv])):
        model_site = []
        model_site.append(data[tv][i]["shop"][0])
        model_site.append(data[tv][i]["title"])
        model_site.append(" ".join(list(data[tv][i]["featuresMap"].values())))
        model.append(model_site)
    datalist.append(model)
data = datalist.copy()
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

#%% Find Model Words from features
model_words = []
for tv in data:
    for i in range(len(tv)):  # ith site on which the tv is sold
        site = tv[i]  # list 0: store {a,n,b}, 1: title, 2: features
        # cleaning titles and features
        title_split = site[1].split()
        for title_part in title_split:
            if re.search(
                "([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)",
                title_part,
            ):
                model_words.append(title_part)

            # if (
            #     len(
            #         re.findall(
            #             "([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)",
            #             title_part,
            #         )
            #     )
            #     != 0
            # ):
            #     model_words.append(title_part)
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
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header
df["mw"] = model_words
df = df.set_index("mw")
a = df.T
b = z_fours[0]
A = df[
    [
        b + "_a",
        b + "_b",
        b + "_n",
    ]
]
B = A[A.sum(axis=1) != 0]


#%% oldies
# all_strings = []
# for i in data:  # i is model key
#     for j in data[i]:  # j is dict that represents one site
#         features_values_list = list(j["featuresMap"].values())
#         for k in features_values_list:
#             to_append = k.split()
#         all_strings.extend(to_append)
# result = list(set(all_strings))

# for tv in data:
#     for i in range(len(data[tv])):  # ith site on which the tv is sold
#         site = data[tv][i]
#         # title cleaning
#         site["title"] = site["title"].lower()
#         site["title"] = site["title"].replace("/", " ")
#         site["title"] = site["title"].replace("-", " ")
#         if any(
#             map(
#                 site["title"].__contains__,
#                 z_punctuations,
#             )
#         ):
#             for punctuation in z_punctuations:
#                 site["title"] = site["title"].replace(punctuation, "")
#         # Standardize inch notation
#         if any(map(site["title"].__contains__, z_inches)):
#             for inch in z_inches:
#                 site["title"] = site["title"].replace(inch, "inch")
#         # Standardize pounds notation
#         if any(map(site["title"].__contains__, z_pounds)):
#             for lbs in z_pounds:
#                 site["title"].replace(lbs, "lb")
#         # Standardize hertz notation
#         if any(map(site["title"].__contains__, z_hertz)):
#             for hz in z_hertz:
#                 site["title"].replace(hz, "hz")

#         # featureMap cleaning
#         for feature in site["featuresMap"]:
#             site["featuresMap"][feature] = site["featuresMap"][feature].lower()
#             # feature_value = site["featuresMap"][feature]
#             data[tv][i]["featuresMap"][feature] = data[tv][i]["featuresMap"][
#                 feature
#             ].replace("/", " ")
#             data[tv][i]["featuresMap"][feature] = data[tv][i]["featuresMap"][
#                 feature
#             ].replace("-", " ")

#             if any(
#                 map(
#                     data[tv][i]["featuresMap"][feature].__contains__,
#                     z_punctuations,
#                 )
#             ):
#                 for punctuation in z_punctuations:
#                     data[tv][i]["featuresMap"][feature] = data[tv][i][
#                         "featuresMap"
#                     ][feature].replace(punctuation, "")
#             # Standardize inch notation
#             if any(
#                 map(data[tv][i]["featuresMap"][feature].__contains__, z_inches)
#             ):
#                 for inch in z_inches:
#                     data[tv][i]["featuresMap"][feature] = data[tv][i][
#                         "featuresMap"
#                     ][feature].replace(inch, "inch")
#             # Standardize pounds notation
#             if any(
#                 map(data[tv][i]["featuresMap"][feature].__contains__, z_pounds)
#             ):
#                 for lbs in z_pounds:
#                     data[tv][i]["featuresMap"][feature] = data[tv][i][
#                         "featuresMap"
#                     ][feature].replace(lbs, "lb")
#             # Standardize hertz notation
#             if any(
#                 map(data[tv][i]["featuresMap"][feature].__contains__, z_hertz)
#             ):
#                 for hz in z_hertz:
#                     data[tv][i]["featuresMap"][feature] = data[tv][i][
#                         "featuresMap"
#                     ][feature].replace(hz, "hz")


# all_strings_fm = []
# for i in data:  # i is model key
#     for j in data[i]:  # j is dict that represents one site
#         features_values_list = list(j["featuresMap"].values())
#         features_values_list_lower = [x for x in features_values_list]
#         all_strings_fm.extend(features_values_list_lower)
# all_strings__fm_split = []
# for i in all_strings_fm:
#     single_words = i.split()
#     for single_word in single_words:
#         all_strings__fm_split.append(single_word)

# z_inch_extracted = [
#     i for i in all_strings__fm_split if (re.findall("inch$", i))
# ]
# z_inch_extracted = list(set(z_inch_extracted))
# # z_inch = []
# # for i in z_inch_extracted:
# #     try:
# #         z_inch.append(str(round(float(i[:-4])))+"inch")
# #     except:
# #         pass
# # z_inch = list(set(z_inch))

# z_hz_extracted = [i for i in all_strings__fm_split if (re.findall("hz$", i))]
# z_hz_extracted = list(set(z_hz_extracted))

# z_lb_extracted = [i for i in all_strings__fm_split if (re.findall("lb$", i))]
# z_lb_extracted = list(set(z_lb_extracted))
# # z_lb = []
# # for i in z_lb_extracted:
# #     try:
# #         z_lb.append(str(round(float(i[:-2])))+"lb")
# #     except:
# #         pass
# # z_lb = list(set(z_lb))

# model_words = z_inch_extracted.copy()
# model_words.extend(z_hz_extracted)
# model_words.extend(z_lb_extracted)

#%% Find Model Words from title
# all_titles = []
# for i in data:  # i is model key
#     for j in data[i]:  # j is dict that represents one site
#         title = j["title"]
#         title_split = title.split()
#         all_titles.extend(title_split)
# for i in all_titles:
#     if (
#         len(
#             re.findall(
#                 "([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)",
#                 i,
#             )
#         )
#         != 0
#     ):
#         model_words.append(i)
# model_words = list(set(model_words))

# (ˆ\d+(\.\d+)?[a-zA-Z]+$|ˆ\d+(\.\d+)?$)
# ([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)

# tv = "47LW5600"
# i = 2
# model_word = '47inch'
# a = ["47G2", "UN46ES6580", "32SL410U"]

for tv in data:
    for i in range(len(data[tv])):  # ith site on which the tv is sold
        site = data[tv][i]
        title = site["title"]
        vector = [str(tv + "_{}".format(site["shop"][0]))]
        features = list(site["featuresMap"].values())
        for model_word in model_words:
            if (model_word in features) or (model_word in title):
                vector.append(1)
            else:
                vector.append(0)
        vectors.append(vector)
#%% MinHash
# kmeans = KMeans(n_clusters=1262, random_state=0).fit(a).labels_
# dbscan = DBSCAN(eps=2).fit(a).labels_


# for i in all_strings_split:
#     if (
#         len(
#             re.findall(
#                 "([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)",
#                 i,
#             )
#         )
#         != 0
#     ):
#         model_words.append(i)
#     elif len(re.findall("(^\d+(\.\d+)?[a-zA-Z]+$|^\d+(\.\d+)?$)", i)) != 0:
#         model_words.append(i)
# model_words = list(set(model_words))
