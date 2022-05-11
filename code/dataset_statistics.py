from util.localcache.huggingface import cache as HF_CACHE
HF_CACHE.enable()
from util.localcache.nltk import cache as NLTK_CACHE
NLTK_CACHE.enable()

import json
import csv
import re
import itertools
import statistics
from pathlib import Path
from collections import Counter
from sys import getsizeof

try:
    import nltk
    nltk.download("punkt")
    word_tokenize = nltk.tokenize.word_tokenize
except:
    def word_tokenize(text):
        return text.split()

try:
    from tqdm import tqdm
except:
    def tqdm(iterable, **kwargs):
        return iterable

DATASETS_AVAILABLE = True
try:
    from datasets import load_dataset
except ImportError:
    print("The `datasets` library is required for loading external datasets such as XSum.")
    print("As those are only used here to compute statistics present in the thesis, the library is not included as a requirement.")
    print("Install with: pip3 install datasets==2.0.0")
    print("More information here: https://huggingface.co/docs/datasets/index")
    DATASETS_AVAILABLE = False

STATISTICS_OUTPUT_PATH = None
VOCABULARY_OUTPUT_PATH = None

# See https://stackoverflow.com/a/7594052
CASE_SPLITTING_PATTERN = re.compile(r"(?<!^)(?<![A-Z])(?=[A-Z])|(?<!^)(?=[A-Z][a-z])")
TOKEN_PATTERN = re.compile(r"\b\w{2,}\b")

# See https://github.com/igorbrigadir/stopwords
SPACY_STOPWORDS = set(["a", "about", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "computer", "con", "could", "couldnt", "cry", "de", "describe", "detail", "did", "didn", "do", "does", "doesn", "doing", "don", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "just", "keep", "kg", "km", "last", "latter", "latterly", "least", "less", "ltd", "made", "make", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps", "please", "put", "quite", "rather", "rather", "re", "really", "regarding", "same", "say", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "unless", "until", "up", "upon", "us", "used", "using", "various", "very", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"])

datasets = {
    "Hadoop": {"message_key": "SimplifiedMessage", "instance_paths": (Path(__file__).parent / ".." / "data" / "Hadoop" / "processed" / "log_instances").glob("*.csv")},
    "TelcoApp": {"message_key": "SimplifiedMessage", "instance_paths": (Path(__file__).parent / ".." / "data" / "TelcoApp" / "processed" / "with_RCA" / "log_instances").glob("*.csv")},
    "LogSummary": {"message_key": "SimplifiedMessage", "instance_paths": (Path(__file__).parent / ".." / "data" / "LogSummary" / "processed" / "log_instances").glob("*.csv")},
}
external_datasets = {
    "CNN/DailyMail": {"message_key": "article", "summary_key": "highlights", "path": "ccdv/cnn_dailymail", "name": "3.0.0", "keep_in_memory": False},
    "XSum": {"message_key": "document", "summary_key": "summary", "path": "xsum", "keep_in_memory": False},
    "AESLC": {"message_key": "email_body", "summary_key": "subject_line", "path": "aeslc", "keep_in_memory": False},
    "BigPatent": {"message_key": "description", "summary_key": "abstract", "path": "big_patent", "name": "all", "keep_in_memory": False},
}

if DATASETS_AVAILABLE:
    datasets.update(external_datasets)

def build_vocabulary(corpus):
    tokenized_sentences = map(TOKEN_PATTERN.findall, corpus)
    tokens = itertools.chain.from_iterable(tokenized_sentences)
    tokens = itertools.chain.from_iterable(map(CASE_SPLITTING_PATTERN.split, tokens))
    tokens = map(lambda x: x.lower(), tokens)
    return Counter(tokens)

def collect_messages_from_log_instances(instance_paths, message_key):
    for path in tqdm(tuple(instance_paths), desc="Reading log instances"):
        with path.open("r") as f:
            for entry in csv.DictReader(f):
                yield entry[message_key]

def most_frequent_words(vocab, n=None, min_frequency=None, percentile=None, ignored_words=SPACY_STOPWORDS):
    most_frequent = vocab.most_common(None)
    if ignored_words:
        most_frequent = filter(lambda x: x[0] not in ignored_words, most_frequent)
    if n:
        most_frequent = itertools.islice(most_frequent, n)
    if min_frequency:
        most_frequent = itertools.takewhile(lambda x: x[1] >= min_frequency, most_frequent)
    if percentile:
        threshold = sum(vocab.values()) * percentile
        accumulated = itertools.accumulate(most_frequent, lambda acc, new: (new[0], acc[1] + new[1]))
        most_frequent = itertools.takewhile(lambda x: x[1] <= threshold, accumulated)
    return set(map(lambda x: x[0], most_frequent))

def vocabulary_overlap_coefficient(vocab_a, vocab_b, **kwargs):
    a = most_frequent_words(vocab_a, **kwargs)
    b = most_frequent_words(vocab_b, **kwargs)
    return len(a.intersection(b)) / len(a.union(b))

def compute_statistics(numbers):
    stats = {}
    stats["mean"] = statistics.mean(numbers)
    stats["std"] = statistics.pstdev(numbers, mu=stats["mean"])
    stats["median"] = statistics.median(numbers)
    return stats

if __name__ == "__main__":
    dataset_stats = {name: {} for name in datasets.keys()}

    for name, info in datasets.items():
        print(">>>", name, "<<<")
        if "instance_paths" in info:
            messages = collect_messages_from_log_instances(info["instance_paths"], info["message_key"])
        else:
            kwargs = info.copy()
            message_key = kwargs.pop("message_key")
            summary_key = kwargs.pop("summary_key", None)
            def process_entries():
                message_lengths = []
                if summary_key:
                    summary_lengths = []
                dataset = load_dataset(**kwargs)
                if isinstance(dataset, dict):
                    dataset = itertools.chain.from_iterable(dataset.values())
                for entry in dataset:
                    message_lengths.append(len(word_tokenize(entry[message_key])))
                    if summary_key:
                        summary_lengths.append(len(word_tokenize(entry[summary_key])))
                    yield entry[message_key]
                stats = compute_statistics(message_lengths)
                dataset_stats[name]["words_per_document"] = stats
                print("Mean words per document: {mean:.3f} ± {std:.3f} (median {median})".format(**stats))
                if summary_key:
                    stats = compute_statistics(summary_lengths)
                    dataset_stats[name]["words_per_summary"] = stats
                    print("Mean words per summary: {mean:.3f} ± {std:.3f} (median {median})".format(**stats))
            messages = process_entries()
        vocab = build_vocabulary(tqdm(messages, desc="Building vocabulary for {}".format(name)))
        info["vocabulary"] = vocab
        vocab_stats = {}
        vocab_stats["size"] = len(vocab)
        vocab_stats["memory_mib"] = getsizeof(vocab) / 1024 / 1024
        vocab_stats["size_no_stopwords"] = len(set(vocab.keys()).difference(SPACY_STOPWORDS))
        dataset_stats[name]["vocabulary"] = vocab_stats
        print("Vocabulary size: {size} ({memory_mib:.3f} MiB); without stopwords: {size_no_stopwords}".format(**vocab_stats))
        print("Top 10 most common tokens:", most_frequent_words(vocab, n=10))

    dataset_stats["overlap"] = {"25%": {}, "500": {}, "10k": {}}

    for x, y in itertools.combinations(datasets.items(), 2):
        overlap = vocabulary_overlap_coefficient(x[1]["vocabulary"], y[1]["vocabulary"], percentile=0.25)
        dataset_stats["overlap"]["25%"][x[0] + ":" + y[0]] = overlap
        print("Overlap of top 25% most common tokens between {} and {}: {:.6f}".format(x[0], y[0], overlap))
        overlap = vocabulary_overlap_coefficient(x[1]["vocabulary"], y[1]["vocabulary"], n=500, min_frequency=3)
        dataset_stats["overlap"]["500"][x[0] + ":" + y[0]] = overlap
        print("Overlap of top 500 most common tokens between {} and {}: {:.6f}".format(x[0], y[0], overlap))
        overlap = vocabulary_overlap_coefficient(x[1]["vocabulary"], y[1]["vocabulary"], n=10000, min_frequency=3)
        dataset_stats["overlap"]["10k"][x[0] + ":" + y[0]] = overlap
        print("Overlap of top 10000 most common tokens between {} and {}: {:.6f}".format(x[0], y[0], overlap))

    if STATISTICS_OUTPUT_PATH:
        with open(STATISTICS_OUTPUT_PATH, "w") as f:
            json.dump(dataset_stats, f, indent=4, sort_keys=True)

    if VOCABULARY_OUTPUT_PATH:
        with open(VOCABULARY_OUTPUT_PATH, "w") as f:
            json.dump(datasets, f, indent=4, sort_keys=True, default=lambda x: list(iter(x)))
