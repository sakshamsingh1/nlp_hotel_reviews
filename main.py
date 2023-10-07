import math
from collections import defaultdict, Counter

class NgramModel:
    def __init__(self, ngram=2, smoothing=None, unknown_handle=None, threshold=None, k=None, top_k_vocab=None):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.threshold = threshold
        self.k = k
        self.ngram = ngram
        self.smoothing = smoothing
        self.top_k_vocab = top_k_vocab
        self.unknown_handle = unknown_handle
        self.epsilon = 1e-10

    def preprocess(self, text):
        text = text.lower()
        text = ' '.join(['<NUM>' if word.isdigit() else word for word in text.split()])
        return text

    def train(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = self.preprocess(line).strip().split()
                tokens_with_boundaries = ["<START>"] + tokens + ["<END>"]
                self.unigram_counts.update(tokens_with_boundaries)
        if self.unknown_handle is not None:
            low_freq_words = []
            if self.unknown_handle == "threshold":
                low_freq_words = [word for word, count in self.unigram_counts.items() if count <= self.threshold]
            elif self.unknown_handle == "top_k_vocab":
                top_k = sorted(self.unigram_counts.items(), key=lambda x: x[1], reverse=True)[:self.top_k_vocab]
                low_freq_words = [word for word, count in self.unigram_counts.items() if word not in top_k]

            for word in low_freq_words:
                self.unigram_counts["<UNK>"] += self.unigram_counts[word]
                del self.unigram_counts[word]

        self.vocab = set(self.unigram_counts.keys())

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = ["<START>"] + [word if word in self.vocab else "<UNK>" for word in self.preprocess(line).strip().split()] + ["<END>"]
                for i in range(1, len(tokens)):
                    self.bigram_counts[tokens[i-1]][tokens[i]] += 1

    def unigram_probability(self, token):
        token = self.preprocess(token)
        if token not in self.vocab:
            token = "<UNK>"

        total = sum(self.unigram_counts.values())
        if self.smoothing is not None:
            if self.smoothing == "add-k":
                return (self.unigram_counts[token] + self.k) / (total + len(self.vocab) * self.k)
            elif self.smoothing == "stupid_backoff":
                return self.unigram_counts[token] / total if self.unigram_counts[token] else self.epsilon
        return self.unigram_counts[token] / total

    def bigram_probability(self, token1, token2):
        token1, token2 = self.preprocess(token1), self.preprocess(token2)
        if token1 not in self.vocab:
            token1 = "<UNK>"
        if token2 not in self.vocab:
            token2 = "<UNK>"

        if self.smoothing is not None:
            if self.smoothing == "add-k":
                return (self.bigram_counts[token1][token2] + self.k) / (self.unigram_counts[token1] + len(self.vocab) * self.k)
            elif self.smoothing == "stupid_backoff":
                if token1 in self.bigram_counts and token2 in self.bigram_counts[token1]:
                    return self.bigram_counts[token1][token2] / self.unigram_counts[token1]
                elif token2 in self.unigram_counts:
                    return 0.4 * self.unigram_counts[token2] / sum(self.unigram_counts.values())
                else:
                    return self.epsilon
        return self.bigram_counts[token1][token2] / self.unigram_counts[token1]

    def perplexity(self, filepath):
        log_sum = 0
        N = 0
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = ["<START>"] + [word if word in self.vocab else "<UNK>" for word in self.preprocess(line).strip().split()] + ["<END>"]
                N += len(tokens)
                for i in range(1, len(tokens)):
                    if self.ngram == 1:
                        prob = self.unigram_probability(tokens[i])
                    else:
                        prob = self.bigram_probability(tokens[i-1], tokens[i])
                    log_sum -= math.log(prob)
        return math.exp(log_sum / N)

def main():
    models = {
        "Unigram Unsmooth": NgramModel(ngram=1, smoothing=None, unknown_handle=None, threshold=None, k=None, top_k_vocab=None),
        "Unigram Smooth with Laplace": NgramModel(ngram=1, smoothing='add-k', unknown_handle=None, threshold=None, k=1, top_k_vocab=None),
        "Unigram Smooth with stupid_backoff": NgramModel(ngram=1, smoothing='stupid_backoff', unknown_handle=None, threshold=None, k=None, top_k_vocab=None),

        "Bigram Unsmooth": NgramModel(ngram=2, smoothing=None, unknown_handle=None, threshold=None, k=None, top_k_vocab=None),
        "Bigram Smooth with Laplace": NgramModel(ngram=2, smoothing='add-k', unknown_handle=None, threshold=None, k=1, top_k_vocab=None),
        "Bigram Smooth with stupid_backoff": NgramModel(ngram=2, smoothing='stupid_backoff', unknown_handle=None, threshold=None, k=None, top_k_vocab=None),
    }

    for name, model in models.items():
        model.train("train.txt")
        print(f"---- {name} ----")
        if model.ngram == 1:
            print("Unigram Probability of 'the':", model.unigram_probability("the"))
        else:
            print("Bigram Probability of ('the', 'students'):", model.bigram_probability("the", "students"))
        if model.smoothing is not None:
            print("Perplexity on validation set:", model.perplexity("val.txt"))
        print()

if __name__ == "__main__":
    main()
