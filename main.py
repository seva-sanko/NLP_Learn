# Заготовка для выполнения в Colab
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "Cats and dogs are pets."
]


def processed_sentence(sentences):
    processed = []
    all_words = set()

    for sentence in sentences:
        sentence = sentence.lower().replace(".", "")
        sentence = sentence.replace("cats", "cat").replace("dogs", "dog")
        words = sentence.split()
        processed.append(words)

        for word in words:
            all_words.add(word)

    vocabulary = sorted(list(all_words))

    return processed, vocabulary

processed_sentences, vocabulary = processed_sentence(sentences)

print("Словарь (vocabulary):", vocabulary)
print("\nОбработанные предложения:")
for i, sent in enumerate(processed_sentences):
    print(f"Предложение {i+1}: {sent}")


def create_bow_vectors(sentences, vocabulary):
    vectors = []

    for sentence in sentences:
        vector = [0] * len(vocabulary)

        for word in sentence:
            if word in vocabulary:
                idx = vocabulary.index(word)
                vector[idx] += 1

        vectors.append(vector)

    return np.array(vectors)


bow_vectors = create_bow_vectors(processed_sentences, vocabulary)

print("\nBoW вектора:")
for i, vec in enumerate(bow_vectors):
    print(f"Предложение {i + 1}: {vec}")


def calculate_cosine_similarity(vectors):
    similarities = cosine_similarity(vectors)
    return similarities

similarity_matrix = calculate_cosine_similarity(bow_vectors)

print("\nМатрица косинусной близости:")
print(similarity_matrix)

print("\nКосинусная близость между предложениями:")
print(f"Предложение 1 и Предложение 2: {similarity_matrix[0][1]:.3f}")
print(f"Предложение 1 и Предложение 3: {similarity_matrix[0][2]:.3f}")
print(f"Предложение 2 и Предложение 3: {similarity_matrix[1][2]:.3f}")

max_similarity = 0
max_pair = (0, 0)
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        if similarity_matrix[i][j] > max_similarity:
            max_similarity = similarity_matrix[i][j]
            max_pair = (i, j)

print(f"Наиболее похожие предложения: {max_pair[0]+1} и {max_pair[1]+1}")
print(f"Косинусная близость: {max_similarity:.3f}")