from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pandas as pd

def get_most_relevant_sentences(article_text, embeddings_model, top_n_perc = .2):
    """
    Extract only the "top_n_perc" sentences of a text "article_text".
    Top n sentences are considered the sentences that are most similar to the whole text.
    Steps:
     1. The embedding of the whole text is computed. 
     2. The whole text is broken into individual sentences.
     3 Embeddings of each individual sentence are computed.
     4. Cosine similarity of each individual sentences against the embedding of the whole text is computed.
     5. Get top n sentences (sentences that more closely resembele the idea of the whole text).
    Receives:
     - article_text: str: The text of a news article.
     - embeddings_model: object: The model that will be used for embeddings. Model should have a .encode functionality to compute the embeddings.
    Returns:
     - Text containing only the top n % most representative sentences of a text.
    """
    # compute embedding of the whole text
    whole_text_embedding = embeddings_model.encode(article_text, show_progress_bar=False)
    # break text in sentences
    sentences = nltk.sent_tokenize(article_text)
    # store the sentences in a DataFrame
    sentences_df = pd.DataFrame(sentences, columns=['sentence'])
    # compute embeddings of each sentence individually
    sentences_embeddings = embeddings_model.encode(sentences, show_progress_bar=False)
     
    # compute cosine similarities of the whole text vs each individual sentence
    cosine_sims = cosine_similarity(
        whole_text_embedding.reshape(1, -1), 
        sentences_embeddings
    )
    # store cosine similarities on a column of the DataFrame
    sentences_df['similarity'] = cosine_sims[0]
    sentences_df.reset_index(inplace=True)
    # n sentences tied to top_n_perc of the article
    top_n = round(len(sentences)*top_n_perc)
    # Top n percent sentences that capture the main idea, sorted by how they appear in the text
    most_relevant_sentences = sentences_df.sort_values(
        # sort by similarity
        by='similarity', 
        # most similars at the top
        ascending=False
    ).head(
        # top 20%
        top_n
    ).sort_values(
        # sort them back by how they appear in the original text
        by='index'
    )[
        # get senteces
        'sentence'
    ].values.tolist() # to python list
    return ' '.join(most_relevant_sentences)

def print_progress_bar(iteration, total, bar_length=50):
    progress = float(iteration) / float(total)
    arrow = '=' * int(round(progress * bar_length) - 1)
    spaces = ' ' * (bar_length - len(arrow))

    print(f'Progress: [{arrow + spaces}] {int(progress * 100)}%', end='\r')

def options_to_numerated_list(options_list):
    options_str = ""
    for i, o in enumerate(options_list):
        options_str += '{0}. {1} \n'.format(i+1, o)
    return options_str