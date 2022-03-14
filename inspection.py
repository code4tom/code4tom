from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import openai


def inspect_cluster(df, inspect_cluster, samples):
    """
    Takes a cluster_id and number of samples to generate a wordcloud and prints samples
    """
    if samples < 0: 
        print("samples must be greater than 0.")
        raise ValueError

    # wordcloud
    wordcloud_terms = df.loc[df["cluster"] == inspect_cluster]
    text = " ".join(words for words in wordcloud_terms.tokens.astype(str))
    wordcloud = WordCloud(background_color="white", width=800, height=400, max_words=60).generate(text)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()

    # prints sentences and tokens
    cluster_df = df[df.cluster == inspect_cluster].sample(samples, replace=True)
    examples = list(set(cluster_df.index))
    inspection_df = df.iloc[examples]
    for i, item in inspection_df.iterrows():
        print(f"Text {i}:\t",item.Todo,f"\nTokens {i}:\t",item.tokens,"\n")


def use_gpt3(df:pd.DataFrame, min_doc_per_cluster:int, random_state=None, key=None, api_base=None) -> pd.DataFrame: 
    """
    min_doc_per_cluster is a int number which acts as a threshold when a cluster counts as a trash cluster
    """
    openai.api_key = key 
    openai.api_base = api_base 
    if min_doc_per_cluster < 1: 
        print("min_doc_per_cluster must be bigger than 1.")
        raise ValueError

    ok_clusters = []

    for i_cluster in set(list(df.cluster)):
        count = len(df[df.cluster == i_cluster])
        if count <= min_doc_per_cluster:
            df[df.cluster==i_cluster].cluster = [-1]*len(df[df.cluster==i_cluster])
            print(f"Cluster {i_cluster} is a thrash cluster")
        else: 
            ok_clusters.append(i_cluster)

    for i_cluster in ok_clusters:
        print(f"Cluster {i_cluster} Topic:", end=" ")
        
        tokenlist = df[df.cluster == i_cluster]["tokens"].map(lambda x: x[:min(1000, len(df[df.cluster == i_cluster])-1)]).sample(min_doc_per_cluster, random_state=random_state)
        str_list = []
        for item in tokenlist: 
            str_list.append(" ".join(item))

        docs = "\n".join(str_list)
        response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt=f"What do the following documents have in common?\n\nDocuments:\n\"\"\"\n{docs}\n\"\"\"\n\nTopic:", 
            temperature=0,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print(response["choices"][0]["text"].replace('\n',' '))
    return df


def use_goose(df:pd.DataFrame, min_doc_per_cluster:int, random_state = None, key = None, api_base=None) -> pd.DataFrame: 
    """
    min_doc_per_cluster is a int number which acts as a threshold when a cluster counts as a trash cluster
    """
    openai.api_key = key 
    openai.api_base = api_base 
    if min_doc_per_cluster < 1: 
        print("min_doc_per_cluster must be bigger than 1.")
        raise ValueError

    ok_clusters = []
    # check if cluster is useful
    for i_cluster in set(list(df.cluster)):
        count = len(df[df.cluster == i_cluster])
        if count <= min_doc_per_cluster:
            df[df.cluster==i_cluster].cluster = [-1]*len(df[df.cluster==i_cluster])  # assign cluster -1
            print(f"Cluster {i_cluster} is a thrash cluster")
        else: 
            ok_clusters.append(i_cluster)

    for i_cluster in ok_clusters:
        print(f"Cluster {i_cluster} Topic:", end=" ")
        tokenlist = df[df.cluster == i_cluster]["tokens"].map(lambda x: x[:min(1000, len(df[df.cluster == i_cluster])-1)]).sample(min_doc_per_cluster, random_state=random_state)
        
        str_list = []
        for item in tokenlist: 
            str_list.append(" ".join(item))

        docs = "\n".join(str_list)
        response = openai.Completion.create(
            engine="gpt-neo-20b",
            prompt=f"What do the following documents have in common?\n\nDocuments:\n\"\"\"\n{docs}\n\"\"\"\n\nTopic:",
            max_tokens=64,
            stream=False)
        print(response["choices"][0]["text"].replace('\n',' '))

    return df