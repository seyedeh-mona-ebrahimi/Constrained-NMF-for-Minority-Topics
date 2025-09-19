import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from OurAlgorithm import train
import os





def highlight_top_words(document, top_words):
    if isinstance(document, list):  # Check if document is a list
        document = ' '.join(document)  # Join list into a single string
    highlighted = document
    for word in top_words:
        highlighted = highlighted.replace(word, f'*{word}*')  # Highlight the top words
    return highlighted


def kl_divergence(p, q):
    log_q = np.where(q != 0, np.log(q), -1000)
    Ip = np.where(p != 0)
    return np.sum(p[Ip] * (np.log(p[Ip]) - log_q[Ip]))
def jensen_shannon_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def rank_documents_by_custom_js(V, W, H):

    n_topics = H.shape[0]
    n_docs = W.shape[0]
    ranked_docs = {}

    for topic_index in range(n_topics):
        topic_word_distribution = H[topic_index, :]
        js_divergences = np.zeros(n_docs)

        for doc_index in range(n_docs):
            doc_word_distribution = V[doc_index, :].todense()
            doc_word_distribution = np.array(doc_word_distribution).squeeze()
            js_divergences[doc_index] = jensen_shannon_divergence(doc_word_distribution, topic_word_distribution)
        ranked_docs[topic_index] = np.argsort(js_divergences)

    return ranked_docs
def get_topicsss(H, top_words, id2word):
    topic_list = []
    for topic in H:
        words_list = sorted(list(enumerate(topic)), key=lambda x: x[1], reverse=True)
        topk = [tup[0] for tup in words_list[:top_words]]
        topk_proportions = [tup[1] for tup in words_list[:top_words]]
        topk_proportions = [x / sum(topk_proportions) for x in topk_proportions]
        topic_list.append([(id2word[i], prop) for i, prop in zip(topk, topk_proportions)])
    return topic_list

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Topic Modeling Script")
    parser.add_argument('--data_path', type=str, default='./synthetic-data.csv', help="Path to input dataset (CSV format)")
    parser.add_argument('--output_path', type=str, default='./final_output.txt', help="Path to save the output results")
    # Algorithm Parameters
    parser.add_argument('--n_topics', type=int, default=15, help="Number of topics")
    parser.add_argument('--W_max', type=float, default=1e-9, help="Max value for W")
    parser.add_argument('--theta_min', type=float, default=0.4, help="Min value for theta")
    parser.add_argument('--MH_indices', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6,7], help="List of Mental Health indices")
    parser.add_argument('--max_iteration', type=float, default=40, help="maximum iteration of the training")
    # parser.add_argument('--param_name', type=int, default=some_value, help="Description of param_name")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Input file {args.data_path} not found")
    data = pd.read_csv(args.data_path)

    # Preprocessing or additional steps can be included here
    # Example: vectorization using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['Sentence'])  # Assuming your data has a 'text_column'
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

    seed_words =[
        "terapeutti",
        "negatiivisuus",
        "depressio",
        "paniikkikohtaus",
        "terapeutti",
        "tunne",
        "positiivisuus",
        "onnellisuus",
        "motivaatio",
        "tasapaino",
        "selkeys",
        "mielenrauha",
        "itsetunto",
        "itseluottamus",
        "ahdistus",
        "masennus",
        "yksinäisyys",
        "epävarmuus",
        "pelko",
        "stressitaso",
        "stressihäiriö",
        "stressitön",
        "stressihormo",
        "mielenterveysongelma",
        "skitsofreenikko",
        "terapia",
        "psykoterapia",
        "lääkäri",
        "neuvonta",
        "tuki",
        "keskustelu",
        "ystävät",
        "perhe",
        "epätoivo",
        "viha",
        "pelko",
        "häpeä",
        "turhautuminen",
        "ahdistuskohtaus",
        "itku",
        "toipuminen",
        "paraneminen",
        "edistyminen",
        "itsehoito",
        "rentoutuminen",
        "hengitellä",
        "chillaa",
        "voimaantuminen",
        "neuvo",
        "jutella",
        "kipu",
        "hullu",
        "hoito",
        "ilo",
        "ärsyttävä",
        "pelastua",
        "heikkous",
        "huume",
        "lepo",
        "odotus",
        "kriisitila",
        "epäsosiaalinen",
        "yhteisöllisyys",
        "psyko",
        "huolestua",
        "väsymys",
        "neuropsykologia",
        "arvostelukyky",
        "ihmissuhde",
        "terveydellinen",
        "Asperger",
        "kuunnella",
        "itsemurha",
        "käyttäytymishäiriö",
        "autismi",
        "ahdistuneisuushäiriö",
        "paniikki",
        "eristäytyminen",
        "yksinäisyys",
        "adhd",
        "anoreksia",
        "bulimia",
        "mielenterveyshäiriö",
        "hallusinaatio",
        "harhaluulo",
        "luottaa",
        "väärinkäyttö",
        "kannabis",
        "väkivalta",
        "lihavuus",
        "kärsimys",
        "itsetuhoisuus",
        "sairaus",
        "paniikkihäiriö",
        "ahdistuneisuushäiriö",
        "ocd",
        "depressio",
        "tarkkaavaisuushäiriö",
        "syömishäiriöt",
        "skitsofrenia",
        "hallusinaatioita",
        "ptsd",
        "toivoton",
        "huolestunut",
        "surullinen",
        "tukea",
        "apua",
        "uupumus",
        "trauma",
        "paniikkikohtaus",
        "psykoterapia",
        "jooga",
        "meditaatio",
        "psykiatria",
        "sairaalahoito",
        "itsehoito",
        "kriisiapu",
        "kriisipuhelin",
        "tukiryhmä"
        "serotoniinivajaus",
        'addiktio',
        'alakulo',
        'burnout',
        'elämänhallinta',
        'erot']
    seed_indices = [i for i, word in enumerate(tfidf_feature_names) if word in set(seed_words)]
    non_seed_indices = [i for i in range(len(tfidf_feature_names)) if i not in seed_indices]
    # Model training
    W, H = train(tfidf_matrix, args.n_topics, args.MH_indices, args.W_max, non_seed_indices, seed_indices, args.theta_min, args.max_iteration)

    result = {}
    result["topic-word-matrix"] = H
    id2word = {i: word for i, word in enumerate(tfidf_feature_names)}
    result["topics"] = get_topicsss(H, 10, id2word)

    ranked_documents = rank_documents_by_custom_js(tfidf_matrix, W, H)
    # Save the output
    with open(args.output_path, 'w') as file:
        for topic_index in range(args.n_topics):
            document_indices = ranked_documents[topic_index][:10]
            file.write(f"Top 10 documents for Topic #{topic_index}:\n")
            file.write(f"{document_indices}\n")

            top_words = set(word for word, _ in result["topics"][topic_index])
            for doc_index in document_indices:
                highlighted_doc = highlight_top_words(data['Sentence'][doc_index], top_words)
                file.write(f"Document #{doc_index}: {highlighted_doc}\n")
        file.write("\nGenerated Topics and Associated Documents:\n")
        for topic_idx, words in enumerate(result["topics"]):
            topic_words = ' - '.join([f"{word}" for word, _ in words])
            file.write(f"Topic #{topic_idx}: {topic_words}\n")

    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
