import math
from collections import defaultdict, Counter

class NgramModel:
    def __init__(self, threshold=1, k=1, ngram=2, smoothing=False):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()
        self.threshold = threshold
        self.k = k
        self.ngram = ngram
        self.smoothing = smoothing

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

        low_freq_words = [word for word, count in self.unigram_counts.items() if count <= self.threshold]
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
        return (self.unigram_counts[token] + self.k * self.smoothing) / (total + len(self.vocab) * self.k * self.smoothing)

    def bigram_probability(self, token1, token2):
        token1, token2 = self.preprocess(token1), self.preprocess(token2)
        if token1 not in self.vocab:
            token1 = "<UNK>"
        if token2 not in self.vocab:
            token2 = "<UNK>"

        if self.smoothing:
            return (self.bigram_counts[token1][token2] + self.k * self.smoothing) / (self.unigram_counts[token1] + len(self.vocab) * self.k * self.smoothing)
        else:
            prob = self.bigram_counts[token1][token2] / self.unigram_counts[token1] if self.unigram_counts[token1] else 0
            return prob if prob > 0 else 1e-10

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
        "Unigram Unsmooth": NgramModel(threshold=1, ngram=1, smoothing=False),
        "Unigram Smooth with Laplace": NgramModel(threshold=1, ngram=1, k=1, smoothing=True),
        "Unigram Smooth with Add-2": NgramModel(threshold=1, ngram=1, k=2, smoothing=True),
        "Bigram Unsmooth": NgramModel(threshold=1, ngram=2, smoothing=False),
        "Bigram Smooth with Laplace": NgramModel(threshold=1, ngram=2, k=1, smoothing=True),
        "Bigram Smooth with Add-2": NgramModel(threshold=1, ngram=2, k=2, smoothing=True),
    }

    for name, model in models.items():
        model.train("train.txt")
        print(f"---- {name} ----")
        if model.ngram == 1:
            print("Unigram Probability of 'the':", model.unigram_probability("the"))
        else:
            print("Bigram Probability of ('the', 'students'):", model.bigram_probability("the", "students"))
        print("Perplexity on validation set:", model.perplexity("val.txt"))
        print()

if __name__ == "__main__":
    main()
