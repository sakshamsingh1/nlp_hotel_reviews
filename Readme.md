Unigram and Bi-gram model implementation for NLP course
```
## Usage

NgramModel(ngram=2, smoothing='add-k', unknown_handle="threshold", threshold=1, top_k_vocab=None), where
- ngram: 1 for unigram, 2 for bigram
- smoothing: 'add-k' or 'stupid-backoff'
- unknown_handle: 'threshold' or 'top-k-vocab'
- threshold: threshold for unknown words
- top_k_vocab: top k words in vocab
```
