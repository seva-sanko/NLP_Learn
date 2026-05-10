from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np

categories = ['rec.sport.baseball', 'sci.space', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(subset='all', categories=categories,
                                shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.25, random_state=42
)

print(f"Размер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)
nb_pred = nb_model.predict(X_test_bow)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"Naive Bayes Accuracy: {nb_acc:.4f}")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_bow, y_train)
lr_pred = lr_model.predict(X_test_bow)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")


feature_names = vectorizer.get_feature_names_out()
for i, class_name in enumerate(newsgroups.target_names):
    top_features = np.argsort(lr_model.coef_[i])[-10:]
    print(f"\nТоп-10 признаков для класса '{class_name}':")
    print([feature_names[idx] for idx in top_features])


misclassified_idx = np.where(lr_pred != y_test)[0]

print(f"Всего ошибок: {len(misclassified_idx)}")
print("\nПримеры ошибок (истинный класс -> предсказанный):")
for idx in misclassified_idx[:5]:
    true_label = newsgroups.target_names[y_test[idx]]
    pred_label = newsgroups.target_names[lr_pred[idx]]
    print(f"\nТекст: {X_test[idx][:300]}...")
    print(f"Истинный класс: {true_label}")
    print(f"Предсказанный класс: {pred_label}")
    print("-" * 50)


print("Количество уникальных классов в обучающей выборке:", len(set(y_train)))