"""Prediction of Users based on tweet embeddings"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .models import User
from .twitter import vectorize_tweet


def predict_user(user1_name, user2_name, hypo_tweet_text):
    """
    Determine and return which user is more likely to say a hypothetical tweet.
    Example run: predict_user('elonmusk', 'nasa', 'Tesla cars are rad')
    returns either 0 (user0_name) or 1 (user1_name)
    """
    user1 = User.query.filter(User.name == user1_name).one()
    user2 = User.query.filter(User.name == user2_name).one()
    user1_vects = np.array([tweet.vect for tweet in user1.tweets])
    user2_vects = np.array([tweet.vect for tweet in user2.tweets])
    vects = np.vstack([user1_vects, user2_vects])
    labels = np.concatenate(
        [np.ones(len(user1.tweets)), np.zeros(len(user2.tweets))])
    hypo_tweet_vect = vectorize_tweet(hypo_tweet_text)

    log_reg = LogisticRegression().fit(vects, labels)

    return log_reg.predict(hypo_tweet_vect.reshape(1, -1))