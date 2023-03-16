import numpy as np
import pandas as pd
import os
# sample comment
os.chdir("c:/School/TabularDataScience/archive")
#print(os.getcwd())
ratings_df = pd.read_csv("ratings_small.csv")

#print(ratings_df.shape)

#print(ratings_df.head())
movies_df = pd.read_csv("movies_metadata.csv")
#print(movies_df.head())
movies_df.drop(movies_df.index[19730],inplace=True)
movies_df.drop(movies_df.index[29502],inplace=True)
movies_df.drop(movies_df.index[35585],inplace=True)
movies_df.id = movies_df.id.astype(np.int64)
#print(type(movies_df.id[0]))
#print(ratings_df.movieId.isin(movies_df.id).sum())
ratings_df = pd.merge(ratings_df,movies_df[['title','id']],left_on='movieId',right_on='id')
#print(ratings_df.head())
ratings_df.drop(['timestamp','id'],axis=1,inplace=True)
ratings_df.shape
#print(ratings_df.sample(5))
#print(ratings_df.isnull().sum())
ratings_count = ratings_df.groupby(by="title")['rating'].count().reset_index().rename(columns={'rating':'totalRatings'})[['title','totalRatings']]
#print(ratings_count.shape[0])
#print(len(ratings_df['title'].unique()))
#print(ratings_count.sample(5))
#print(ratings_count.head())
#print(ratings_df.head())
ratings_total = pd.merge(ratings_df,ratings_count,on='title',how='left')
#print(ratings_total.shape)
#print(ratings_total.head())
#print(ratings_count['totalRatings'].describe())
#print(ratings_count['totalRatings'].quantile(np.arange(.6,1,0.01)))
votes_count_threshold = 20
ratings_top = ratings_total.query('totalRatings > @votes_count_threshold')
#print(ratings_top.shape)
#print(ratings_top.head())
if not ratings_top[ratings_top.duplicated(['userId','title'])].empty:
    ratings_top = ratings_top.drop_duplicates(['userId','title'])
#print(ratings_top.shape)
df_for_knn = ratings_top.pivot(index='title',columns='userId',values='rating').fillna(0)
#print(df_for_knn.head())
#print(df_for_knn.shape)
from scipy.sparse import csr_matrix
df_for_knn_sparse = csr_matrix(df_for_knn.values)
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine',algorithm='brute')
print(model_knn.fit(df_for_knn_sparse))
#check line above
query_index = np.random.choice(df_for_knn.shape[0])
distances, indices = model_knn.kneighbors(df_for_knn.loc['Batman Returns'].values.reshape(1,-1),n_neighbors=6)
distances, indices = model_knn.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0,len(distances.flatten())):
    if i==0:
        print("Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))

def encode_units(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
    
df_for_ar = df_for_knn.T.applymap(encode_units)

df_for_ar.shape

df_for_ar.head

df_for_ar = df_for_ar.dropna()
df_for_ar = df_for_ar.fillna(df_for_ar.mean())
from fancyimpute import KNN
df_for_ar = KNN(k=5).fit_transform(df_for_ar)


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df_for_ar = pd.DataFrame(df_for_ar)
frequent_itemsets = apriori(df_for_ar, min_support=0.07, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()






query_index = df_for_knn.index.get_loc('Batman Returns')
print(query_index)

distances, indices = model_knn.kneighbors(df_for_knn.iloc[query_index,:].values.reshape(1,-1),n_neighbors=6)
for i in range(0,len(distances.flatten())):
    if i==0:
        print("KNN Recommendations for movie: {0}\n".format(df_for_knn.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}".format(i,df_for_knn.index[indices.flatten()[i]],distances.flatten()[i]))


# Check if the DataFrame is empty
#if rules.empty:
 #   print('The DataFrame is empty')
#else:
 #   print('The DataFrame is not empty')


# Check if the DataFrame has a column named 'antecedants'
#if 'antecedants' in rules.columns:
 #   print('The DataFrame has a column named "antecedants"')
#else:
 #   print('The DataFrame does not have a column named "antecedants"')
  #  print(rules.columns)

all_antecedents = [list(x) for x in rules['antecedents'].values]
desired_indices = [i for i in range(len(all_antecedents)) if len(all_antecedents[i])==1 and all_antecedents[i][0]=='Batman Returns']
#desired_indices = [i for i in range(len(all_antecedents)) if 'Batman Returns' in all_antecedents[i]]
apriori_recommendations=rules.iloc[desired_indices,].sort_values(by=['lift'],ascending=False)
print("hello world")
#print(rules['antecedents'])
#print(all_antecedents)
#print(desired_indices)
#print(len(all_antecedents))
#print(rules)
#print(rules['antecedents'].values)
#for antecedent_list in all_antecedents:
 #   print(len(antecedent_list))

print(apriori_recommendations.head())
print("hello world")
#apriori_recommendations_list = [list(x) for x in apriori_recommendations['consequents'].values]
#print("Apriori Recommendations for movie: Batman Returns\n")
#for i in range(5):
 #   print("{0}: {1} with lift of {2}".format(i+1,apriori_recommendations_list[i],apriori_recommendations.iloc[i,6]))

#apriori_single_recommendations = apriori_recommendations.iloc[[x for x in range(len(apriori_recommendations_list)) if len(apriori_recommendations_list[x])==1],]
#apriori_single_recommendations_list = [list(x) for x in apriori_single_recommendations['consequents'].values]
#print("Apriori single-movie Recommendations for movie: Batman Returns\n")
#for i in range(5):
 #   print("{0}: {1}, with lift of {2}".format(i+1,apriori_single_recommendations_list[i][0],apriori_single_recommendations.iloc[i,6]))

#import matplotlib.pyplot as plt

# Scatter plot of support vs. confidence
#plt.scatter(apriori_recommendations['support'], apriori_recommendations['confidence'], alpha=0.5)
#plt.xlabel('support')
#plt.ylabel('confidence')
#plt.title('Association Rules')
#plt.show()

