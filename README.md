# NLP Learn

NLP concepts implemented from scratch for learning — no heavy frameworks, just numpy and scikit-learn.

## Contents

### `main.py` — Bag-of-Words & Cosine Similarity
Manual BoW vectorization pipeline:
- Tokenization and normalization
- Vocabulary construction
- BoW vector creation
- Cosine similarity matrix between sentences

### `NaiveBayes_LogisticRegression.py` — Text Classification
Text classification on the 20 Newsgroups dataset (`rec.sport.baseball`, `sci.space`, `talk.politics.guns`):
- CountVectorizer (BoW, 5000 features)
- Multinomial Naive Bayes
- Logistic Regression
- Accuracy comparison, top-10 features per class, error analysis

### `Embedings.py` — Count-based Word Embeddings from Scratch
Full pipeline without any NLP libraries:
1. Build vocabulary from corpus
2. Co-occurrence matrix (sliding context window)
3. PPMI (Positive Pointwise Mutual Information)
4. SVD dimensionality reduction → dense word vectors
5. Cosine similarity search for nearest neighbors

## Concepts covered

| Concept | File |
|---------|------|
| Bag-of-Words | `main.py`, `NaiveBayes_LogisticRegression.py` |
| TF-IDF | `NaiveBayes_LogisticRegression.py` |
| Naive Bayes | `NaiveBayes_LogisticRegression.py` |
| Logistic Regression | `NaiveBayes_LogisticRegression.py` |
| Co-occurrence matrix | `Embedings.py` |
| PPMI | `Embedings.py` |
| SVD | `Embedings.py` |
| Cosine similarity | `main.py`, `Embedings.py` |

## Requirements

```bash
pip install numpy scikit-learn
```
