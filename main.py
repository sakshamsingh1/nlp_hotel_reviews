import math
from collections import defaultdict, Counter

class NgramModel:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.vocab = set()

    def train(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                tokens = ["<START>"] + line.strip().split() + ["<END>"]
                self.vocab.update(tokens)
                self.unigram_counts.update(tokens)
                for i in range(1, len(tokens)):
                    self.bigram_counts[tokens[i-1]][tokens[i]] += 1

    def unigram_probability(self, token):
        total = sum(self.unigram_counts.values())
        return self.unigram_counts[token] / total

    def bigram_probability(self, token1, token2):
        return self.bigram_counts[token1][token2] / self.unigram_counts[token1]


class NgramModelWithSmoothing(NgramModel):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def unigram_probability(self, token):
        total = sum(self.unigram_counts.values()) + len(self.vocab)*self.k
        return (self.unigram_counts[token] + self.k) / total

    def bigram_probability(self, token1, token2):
        return (self.bigram_counts[token1][token2] + self.k) / (self.unigram_counts[token1] + len(self.vocab)*self.k)


class NgramModelWithPerplexity(NgramModelWithSmoothing):
    def perplexity(self, filepath):
        log_sum = 0
        N = 0
        with open(filepath, 'r') as file:
            for line in file:
                tokens = ["<START>"] + line.strip().split() + ["<END>"]
                N += len(tokens)
                for i in range(1, len(tokens)):
                    prob = self.bigram_probability(tokens[i-1], tokens[i])
                    log_sum -= math.log(prob)
        return math.exp(log_sum / N)


# Usage
model = NgramModelWithPerplexity(k=1)
model.train("train.txt")
print("Unigram Probability of 'the':", model.unigram_probability("the"))
print("Bigram Probability of ('the', 'students'):", model.bigram_probability("the", "students"))
print("Perplexity on validation set:", model.perplexity("val.txt"))
