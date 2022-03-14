
<div align="center" style="background-color: rgb(78,204,163)">
    <img src = "res/PrelimLogo.png">
</div>



=======

# Table of Contents

[TOC]

# Features

- Add features 
- More features
- Even more features



# Installation
(not yet supported)
To install this package use: 

```
$ pip install code4tom
```


## Quick start

Start with loading your data using pandas. Next, rename the column which you want to work with as "Todo" as shown in the example below.

```python
df = pd.read_csv("data/covidtweets.csv", sep=";")
name_of_text_row = "text"
df = df[[name_of_text_row]]
df["Todo"] = df[name_of_text_row].astype(str)
```



Continue by tokenizing your data. 

```python
df["tokens"] = tokenize(df)
# remove samples where only a few tokens survived - in this case at least > 2 
df = df[df['tokens'].map(lambda d: len(d)) > 2]
df = df.reset_index(drop=True)
```



Choose if you want to use Yake! Keywords or POS tagging to obtain a list of words used for the FastText model.

```python
toplist = get_toplist(df,use_yake=True)
```



Generate your model. 

```python 
model = generate_model(toplist)
```



Calculate the row vectors. 

```python
rowvectors = []
for item in toplist:
    rowvectors.append(documentembedding(item, model)) 

df["rowvectors"] = rowvectors

# furthermore, save rowvectors as a df_vec
df_vec = pd.DataFrame(rowvectors)
df_vec = df_vec.dropna()
```



Before applying clustering to the data, normalize it using PCA.  

```python
df_vec = normalize(df_vec, norm='l2')
pca = decomposition.PCA()
pca.n_components = 20
pca_vec = pca.fit_transform(df_vec)
```



Having obtained PCA vectors, we can now use Louvain clustering on our data.

```python
p,c,G = cluster_louvain(pca_vec)
```



To assess the goodness of fit, use modularity. 

```python
get_modularity(p,G)
```

This value should be >0.5, if this is not the case try different values. 



Now you can start inspecting the clusters found by Louvain clustering.

To only use a few documents, use the function get_document_per_cluster. This function searches the cluster with the least amount of documents and returns a minimum amount of documents you are using for inspection. 

```python
doc_per_cluster = get_document_per_cluster(df_vec, percentage=0.001, maximum_doc_size=10)
```



Inspecting the clusters. (Here cluster with ID 3) 

```python
inspect_cluster(df_vec, 3, doc_per_cluster)
```



To get even more out of your cluster, check out GPT3 or GooseAI

```python
use_gpt3(df_vec, doc_per_cluster)
```

```python
use_goose(df_vec, doc_per_cluster)
```
