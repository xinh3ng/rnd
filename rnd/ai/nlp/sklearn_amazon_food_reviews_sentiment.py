"""
https://www.kaggle.com/eugen1701/predicting-sentiment-and-helpfulness/notebook (sentiment analysis model)
https://www.kaggle.com/laowingkin/amazon-fine-food-review-sentiment-analysis/notebook (personalized food taste)
"""
from pdb import set_trace as debug
import os
import sqlite3
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from pydsutils.generic import create_logger

logger = create_logger(__name__, level="info")
line_width = 120


def add_sentiment_score(data, verbose=1):
    assert sum(data["Score"] == 3) == 0, "Must have no sentiment score equal to 0"
    data["sentiment_cat"] = data["Score"].apply(lambda x: "positive" if x > 3 else "negative")
    data["Usefulness"] = (data["VotesHelpful"] / data["VotesTotal"]).apply(lambda x: "useful" if x > 0.8 else "useless")
    if verbose >= 1:
        logger.info(
            "Successfully gen categorical sentiment scores. Showing 10 random examples:\n%s"
            % data.sample(10).to_string(line_width=line_width)
        )
    return data


def cleanup(sentence, cleanup_re=re.compile("[^a-z]+")):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(" ", sentence).strip()
    # sentence = " ".join(nltk.word_tokenize(sentence))
    return sentence


def show_user_summary_stats(data):
    data = data.copy()
    data = data.groupby(["UserId", "ProfileName"]).agg({"Score": ["count", "mean"]})
    data.columns = data.columns.get_level_values(1)
    data.columns = ["Score count", "Score mean"]
    data = data.sort_values(by="Score count", ascending=False)
    logger.info("Showing top 20 users in total reviews:\n%s" % data.head(20).to_string(line_width=line_width))


def gen_x_tfidf(train_data, test_data, verbose=1):
    """Gnerate features with TF-IDF approach

    :param train_data:
    :param test_data:
    :param verbose:
    :return:
    """
    if verbose >= 1:
        logger.info("Started running gen_x_tfidf()...")
    # Convert text documents to a matrix of token counts
    count_vect = CountVectorizer(min_df=1, ngram_range=(1, 4))
    x_train_counts = count_vect.fit_transform(train_data["Summary_Clean"])

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    word_features = count_vect.get_feature_names()
    if verbose >= 1:
        logger.info(
            "Showing first and last 5 word features:\n%s\n%s\n" % (str(word_features[:5]), str(word_features[-5:]))
        )

    # Processing test data
    x_test_counts = count_vect.transform(test_data["Summary_Clean"])
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    if verbose >= 1:
        logger.info("Successfully generated features through tf-idf")
        logger.info("Shape of train and test features: %s, %s" % (str(x_train_tfidf.shape), str(x_test_tfidf.shape)))
    return x_train_tfidf, x_test_tfidf, word_features


def main(val_pct=0.1):
    con = sqlite3.connect("{}/data/amazon-fine-food-reviews/database.sqlite".format(os.environ["HOME"]))
    raw = pd.read_sql_query(
        con=con,
        sql="""
SELECT * 
FROM Reviews 
ORDER BY RANDOM() 
LIMIT 3
""",
    )
    logger.info("Example raw data:\n%s" % raw.to_string(line_width=line_width))
    data = pd.read_sql_query(
        con=con,
        sql="""
SELECT 
  ProductId,
  UserId,    
  ProfileName,
  Score, 
  Summary, 
  HelpfulnessNumerator as VotesHelpful, 
  HelpfulnessDenominator as VotesTotal
FROM Reviews 
WHERE Score != 3  -- score is between 1 and 5. 3 shows no directional emotion
""",
    )

    # Process data
    data = add_sentiment_score(data, verbose=1)
    data["Summary_Clean"] = data["Summary"].apply(cleanup)
    logger.info("Successfully cleaned up Summary column")

    show_user_summary_stats(data)
    train_data, val_data = train_test_split(data, test_size=val_pct)
    y_train, y_val = train_data["sentiment_cat"], val_data["sentiment_cat"]

    # Create feature data
    x_train_tfidf, x_val_tfidf, word_features_train = gen_x_tfidf(train_data, val_data, verbose=1)

    # Train several models
    for name, model in {"bernoullinb": BernoulliNB(), "logit": LogisticRegression(C=1e5)}.items():
        model.fit(x_train_tfidf, y_train)
        logger.info("Successfully fit model: %s" % name)

        predictions = model.predict(x_train_tfidf)
        perf = metrics.classification_report(y_train, predictions, target_names=["positive", "negative"])

        logger.info("In-sample performance:\n%s\n" % perf)

        predictions = model.predict(x_val_tfidf)
        perf = metrics.classification_report(y_val, predictions, target_names=["positive", "negative"])
        logger.info("Out-of-sample performance:\n%s\n" % perf)

    feature_coefs = pd.DataFrame(
        data=list(zip(word_features_train, model.coef_[0])), columns=["feature", "coef"]
    ).sort_values(by="coef")
    logger.info("Showing top 10 negative words:\n%s" % feature_coefs.head(10).to_string(line_width=line_width))
    logger.info("Showing top 10 positive words:\n%s" % feature_coefs.tail(10).to_string(line_width=line_width))
    return


if __name__ == "__main__":
    main()
    logger.info("ALL DONE!\n")
