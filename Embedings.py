"""
Count-based Word Embeddings: Co-occurrence Matrix + PPMI
=========================================================
Реализация с нуля на numpy — без сторонних NLP библиотек.

Шаги:
  1. Строим словарь
  2. Считаем co-occurrence matrix (окно размера L)
  3. Считаем PPMI
  4. Сжимаем через SVD → плотные векторы
  5. Ищем ближайших соседей по косинусному сходству
"""

import numpy as np
from collections import defaultdict

# ─── 0. Корпус ────────────────────────────────────────────────────────────────

corpus = [
    ["рыжий", "кот", "сидит", "на", "коврике"],
    ["серый", "кот", "спит", "на", "диване"],
    ["большой", "пёс", "сидит", "во", "дворе"],
    ["кот", "и", "пёс", "друзья"],
    ["пёс", "спит", "во", "дворе"],
]


# ─── 1. Словарь ───────────────────────────────────────────────────────────────

def build_vocab(corpus):
    """Строим word→index и index→word."""
    words = sorted({word for sentence in corpus for word in sentence})
    w2i = {w: i for i, w in enumerate(words)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


w2i, i2w = build_vocab(corpus)
V = len(w2i)  # размер словаря
print(f"Словарь ({V} слов): {list(w2i.keys())}\n")


# ─── 2. Co-occurrence Matrix ──────────────────────────────────────────────────

def build_cooccurrence(corpus, w2i, window=2):
    """
    Для каждого слова считаем, сколько раз рядом (в окне ±window)
    встречается каждое другое слово.

    Матрица M[i, j] = сколько раз слово i встречалось в контексте слова j.
    """
    V = len(w2i)
    M = np.zeros((V, V), dtype=np.float32)

    for sentence in corpus:
        for center_pos, center_word in enumerate(sentence):
            center_idx = w2i[center_word]

            # Берём слова в окне вокруг центрального
            left = max(0, center_pos - window)
            right = min(len(sentence), center_pos + window + 1)

            for context_pos in range(left, right):
                if context_pos == center_pos:
                    continue  # само с собой не считаем
                context_idx = w2i[sentence[context_pos]]
                M[center_idx, context_idx] += 1

    return M


M = build_cooccurrence(corpus, w2i, window=2)

print("Co-occurrence matrix:")
print(f"{'':10}", end="")
for i2w_val in [i2w[i] for i in range(V)]:
    print(f"{i2w_val:8}", end="")
print()
for i in range(V):
    print(f"{i2w[i]:10}", end="")
    for j in range(V):
        print(f"{int(M[i, j]):8}", end="")
    print()
print()


# ─── 3. PPMI ──────────────────────────────────────────────────────────────────

def compute_ppmi(M):
    """
    PPMI(w, c) = max(0,  log2( P(w,c) / (P(w) * P(c)) ))

    P(w, c) = M[w,c] / sum(M)          — вероятность пары
    P(w)    = sum(M[w,:]) / sum(M)      — вероятность слова w
    P(c)    = sum(M[:,c]) / sum(M)      — вероятность контекста c
    """
    total = M.sum()  # общее кол-во пар
    p_wc = M / total  # P(w, c)
    p_w = M.sum(axis=1, keepdims=True) / total  # P(w)  — по строкам
    p_c = M.sum(axis=0, keepdims=True) / total  # P(c)  — по столбцам

    # Считаем PMI; там где p_wc=0 будет -inf → заменим на 0 после
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log2(p_wc / (p_w * p_c))

    pmi[~np.isfinite(pmi)] = 0  # -inf и nan → 0
    ppmi = np.maximum(pmi, 0)  # PPMI = max(PMI, 0)
    return ppmi


PPMI = compute_ppmi(M)

print("PPMI matrix (округлено до 2 знаков):")
print(f"{'':10}", end="")
for i in range(V):
    print(f"{i2w[i]:8}", end="")
print()
for i in range(V):
    print(f"{i2w[i]:10}", end="")
    for j in range(V):
        val = PPMI[i, j]
        print(f"{val:8.2f}", end="")
    print()
print()


# ─── 4. SVD — сжимаем до d измерений ─────────────────────────────────────────

def svd_reduce(matrix, d):
    """
    Берём d главных компонент через SVD.
    matrix ≈ U @ diag(S) @ Vt
    Используем U @ diag(S) как плотные векторы слов.
    """
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    # Берём только первые d компонент
    word_vectors = U[:, :d] * S[:d]
    return word_vectors


d = 3  # маленький корпус → 3 измерения достаточно
vectors = svd_reduce(PPMI, d=d)

print(f"Векторы слов после SVD (d={d}):")
for i in range(V):
    vec_str = "  ".join(f"{x:7.3f}" for x in vectors[i])
    print(f"  {i2w[i]:10} [{vec_str}]")
print()


# ─── 5. Косинусное сходство + ближайшие соседи ───────────────────────────────

def cosine_similarity(v1, v2):
    """cos(θ) = (v1 · v2) / (|v1| * |v2|)"""
    dot = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 0.0
    return dot / norms


def most_similar(word, vectors, w2i, i2w, top_n=3):
    """Находим top_n ближайших слов по косинусному сходству."""
    if word not in w2i:
        print(f"Слово '{word}' не в словаре!")
        return []

    idx = w2i[word]
    query_vec = vectors[idx]

    scores = []
    for i in range(len(vectors)):
        if i == idx:
            continue
        sim = cosine_similarity(query_vec, vectors[i])
        scores.append((i2w[i], sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


print("Ближайшие соседи:")
for query in ["кот", "пёс", "сидит", "спит"]:
    neighbors = most_similar(query, vectors, w2i, i2w, top_n=3)
    neighbors_str = ",  ".join(f"{w} ({s:.2f})" for w, s in neighbors)
    print(f"  {query:10} → {neighbors_str}")
print()

# ─── 6. Матрица попарных сходств ─────────────────────────────────────────────

print("Матрица косинусного сходства между всеми словами:")
words_short = [i2w[i] for i in range(V)]
print(f"{'':10}", end="")
for w in words_short:
    print(f"{w:8}", end="")
print()
for i in range(V):
    print(f"{i2w[i]:10}", end="")
    for j in range(V):
        sim = cosine_similarity(vectors[i], vectors[j])
        print(f"{sim:8.2f}", end="")
    print()

print("\n✓ Готово!")
print("Обрати внимание: 'кот' и 'пёс' должны быть ближе друг к другу,")
print("чем 'кот' и 'коврик' — потому что встречаются в похожих контекстах.")