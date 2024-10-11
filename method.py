import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

import utils


def compute_count_sentiment(token_list, positive_words, negative_words):
    """
        Basic sentiment analysis by counting the number of positive words, counting the negative words.
        The overall polarity is the difference in the two numbers.

        @param token_list: token list from a post + associated comments
        @param positive_words: set of positive sentiment words
        @param negative_words: set of negative sentiment words

        @returns: Difference of the positive and negative word count
    """
    positive_word_count = len([tok for tok in token_list if tok in positive_words])
    negative_word_count = len([tok for tok in token_list if tok in negative_words])

    sentiment = positive_word_count - negative_word_count

    return sentiment


def sentiment_analysis(method, posts_df):
    """
    Analysing the sentiment of the posts and comments via the methods 'Count' and 'Vader'.

    Count sentiment analysis -> Basic sentiment analysis by counting the number of positive words,
    counting the negative words. The overall polarity is the difference in the two numbers.

    Vader sentiment analysis -> Using Vader lexicons for sentiment analysis instead of
    raw positive and negative word counts.

    @param method: 'Vader' or 'Count'
    @param posts_df: The df with post and comment data

    @returns: list of reddit posts' sentiments, in the format of [date, sentiment]
    """
    set_pos_words = []
    set_neg_words = []
    if method == 'Count':
        # load pos, neg word lists
        set_pos_words = utils.read_file('positive-words.txt')
        set_neg_words = utils.read_file('negative-words.txt')

    sentiment_list = []
    vader_sentiment_analyser = SentimentIntensityAnalyzer()

    for row in posts_df.itertuples(index=True):
        print_processing = True if row[0] <= 1 else False

        token_list = row.processed_tokens

        date = row.utc_date
        sentiment = 0

        # compute sentiment
        if method == 'Vader':
            sentiment = vader_sentiment_analyser.polarity_scores(" ".join(token_list))
            sentiment_list.append([pd.to_datetime(date, unit='s'), sentiment['compound']])
        elif method == 'Count':
            sentiment = compute_count_sentiment(token_list, set_pos_words, set_neg_words)
            # save the date and sentiment of each reddit post
            sentiment_list.append([pd.to_datetime(date, unit='s'), sentiment])

        if print_processing:
            title = row.title
            desc = row.desc
            num_comments = row.num_comments
            date = row.formatted_date

            start = '\n\n------------Analysing sentiment------------\n'
            end = '\n------------------------------------\n\n'
            formatted_post = f'Date: {date}\n\nPost title:\n{title}\n\nPost desc:\n{desc}\n\nNum Comments: {num_comments}'

            print(utils.yellow_rgb + start + formatted_post + end, end='')

            utils.print_coloured_tokens(method, token_list, sentiment, set_pos_words, set_neg_words)

    return sentiment_list
