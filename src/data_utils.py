import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_tweet(text):
    """
    Clean a HumAID tweet while preserving the actual tweet content.
    """
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Remove RT @username: prefix when it appears at the beginning
    text = re.sub(r"^RT\s+@\w+:\s*", "", text)

    return text.strip()


def load_and_split(
    csv_or_df,
    test_size=0.3,
    val_size=0.5,
    seed=42
):
    """
    Load HumAID data, clean tweets, remove duplicates,
    and create stratified train/validation/test splits.
    """

    # Accept either a DataFrame or a CSV path
    if isinstance(csv_or_df, pd.DataFrame):
        df = csv_or_df.copy()
    else:
        df = pd.read_csv(csv_or_df)

    # Clean tweets
    df["text_clean"] = df["tweet_text"].apply(clean_tweet)

    # Remove exact duplicate cleaned tweets
    df = df.drop_duplicates(subset="text_clean")

    # 70% train, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["class_label"],
        random_state=seed
    )

    # Split remaining 30% into 15% validation + 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        stratify=temp_df["class_label"],
        random_state=seed
    )

    return train_df, val_df, test_df
