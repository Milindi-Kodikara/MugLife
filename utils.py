# utility functions

# File containing utility functions
import re


def get_color_escape(r, g, b, background=False):
    """
        Combine the r,g,b colour values into the desired format for coloured terminal prints

        @param r: the red colour value
        @param g: the green colour value
        @param b: the blue colour value
        @param background: if the colour is for the background or font colour

        @return: formatted colour string
    """
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)


green = '#275c4d'
red = '#af221d'
yellow = '#c59103'

green_rgb = get_color_escape(39, 92, 77)
red_rgb = get_color_escape(175, 34, 29)
yellow_rgb = get_color_escape(197, 145, 3)
RESET = '\033[0m'


def read_file(filename):
    """
        Read a file

        @param filename: name of the file to read

        @return: a unique list of words
    """
    item_list = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            item_list.append(line.strip())

    return set(item_list)


# The following regex has been created with the help of:
# https://stackoverflow.com/questions/73804264/removing-emojis-and-special-characters-in-python


regex_emojis = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"
                          u"\u3030"
                          "]+", re.UNICODE)


def print_sentiment(sentiment, prefix=''):
    """
        Formatted print of the sentiment value

        @param sentiment: sentiment value
        @param prefix: 'pos', 'neg', 'neu' or 'compound' for Vader sentiment analysis
    """
    start = '\n\n------------Count sentiment value------------\n'
    end = '\n------------------------------------\n\n'
    if sentiment > 0:
        print(green_rgb + start + prefix + str(sentiment) + end, end='')
    elif sentiment < 0:
        print(red_rgb + start + prefix + str(sentiment) + end, end='')
    else:
        print(yellow_rgb + start + prefix + str(sentiment) + end, end='')


def print_coloured_tokens(method, token_list, sentiment, positive_words=None, negative_words=None):
    """
        Formatted print of the tokens based on the sentiment value and colored tokens for count sentiment analysis

        @param method: Sentiment analysis method i.e 'Count' or 'Vader'
        @param token_list: list of tokens extracted from the post + associated comments
        @param sentiment: sentiment value for the post + associated comments based on the tokens
        @param positive_words: set of positive sentiment words
        @param negative_words: set of negative sentiment words
    """
    if positive_words is None:
        positive_words = []
    if method == 'Count':
        for token in token_list:
            if token in positive_words:
                print(green_rgb + token + ', ', end='')
            elif token in negative_words:
                print(red_rgb + token + ', ', end='')
            else:
                print(yellow_rgb + token + ', ', end='')

        print_sentiment(sentiment)

    if method == 'Vader':
        for cat, score in sentiment.items():
            print(*token_list, sep=', ')
            prefix = '{}: '.format(cat)
            print_sentiment(score, prefix)
