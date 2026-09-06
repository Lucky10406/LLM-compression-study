from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_baseline(train_df, val_df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df["text_clean"])
    X_val = vectorizer.transform(val_df["text_clean"])

    y_train = train_df["class_label"]

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    clf.fit(X_train, y_train)

    return clf, vectorizer, X_val
