# helper functions for pre-processing
import os

import networkx as nx
import pandas as pd
from datetime import datetime
import string
from praw.models import MoreComments
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

import method
import re
import utils

tea = 'tea'
coffee = 'coffee'

tokeniser = TweetTokenizer()

# add punctuation to stopwords list
stop_words = stopwords.words('english') + list(string.punctuation) + ['rt', 'via', '...', '…', '"', "'", '`', '-', '..']


def process(text, print_processing=False):
    """
        Perform the processing of the reddit posts
        @param text: the text (reddit post/comment) to process
        @param print_processing: Bool to determine whether to print out the cleaned token list at each step

        @returns: list of (valid) tokens
    """
    start = '\n\n------------------------------------\n'
    end = '\n------------------------------------\n\n'
    if print_processing:
        print(utils.yellow_rgb + f'{start}Initial text\n{text}\n{end}')

    # conversion to lowercase
    text = text.lower()
    if print_processing:
        print(utils.red_rgb + f'{start}Lowercase text{start}{text}\n{end}', end='')

    # remove curly inverted commas
    text = re.sub(u"(\u2018|\u2019|\u2014|\u201C|\u201D)", "", text)

    if print_processing:
        print(utils.red_rgb + f'{start}Inverted comma removed text{start}{text}\n{end}', end='')

    # remove emojis
    text = re.sub(utils.regex_emojis, '', text)
    if print_processing:
        print(utils.red_rgb + f'{start}Emoji removed text{start}{text}\n{end}', end='')

    # remove username tags, mentions, and links
    text = re.sub(r'(r/|@|https?)\S+|#', '', text)
    if print_processing:
        print(utils.red_rgb + f'{start}Tags, mentions and links removed text{start}{text}\n{end}', end='')

    # tokenizer
    tokens = tokeniser.tokenize(text)

    if print_processing:
        print(utils.red_rgb + f'{start}Tokenized text{start}{tokens}\n{end}', end='')

    # strip whitespace
    tokens = [tok.strip() for tok in tokens]

    if print_processing:
        print(utils.red_rgb + f'{start}Whitespace stripped tokenized text{start}{tokens}\n{end}', end='')

    # remove digits
    tokens = [tok for tok in tokens if not tok.isdigit()]
    if print_processing:
        print(utils.red_rgb + f'{start}Digits removed tokenized text{start}{tokens}\n{end}', end='')

    # remove stop words
    tokens = [tok for tok in tokens if tok not in stop_words]

    if print_processing:
        print(utils.red_rgb + f'{start}Stop words removed tokenized text{start}{tokens}\n{end}', end='')

    if print_processing:
        print(utils.green_rgb + f'{start}Final tokenized text{start}{tokens}\n{end}', end='')

    return tokens


def reddit_data_collection(
        data_folder_path,
        collected_posts, data_sample_filepath):
    """
        Collecting data from reddit.

        @param data_folder_path: Folder path to save the reply graph
        @param collected_posts: List of posts collected from reddit
        @param collected_posts:
        @param data_sample_filepath


        @returns: The updated dataframes and lists with reddit data and tokens
    """
    tea_unprocessed_token_lists = []
    tea_processed_token_lists = []

    coffee_unprocessed_token_lists = []
    coffee_processed_token_lists = []

    tea_reply_graph = nx.DiGraph()
    tea_reply_graph_filepath = f'{data_folder_path}/reddit_tea_reply_graph.graphml'

    coffee_reply_graph = nx.DiGraph()
    coffee_reply_graph_filepath = f'{data_folder_path}/reddit_coffee_reply_graph.graphml'

    posts_df = pd.DataFrame(
        columns=['social_media_id', 'post_type', 'title', 'utc_date', 'formatted_date', 'desc', 'author', 'rating',
                 'num_comments', 'unprocessed_tokens', 'processed_tokens'])

    # track the ids of post and comments for the reply graph
    post_comment_ids = dict()

    for post in collected_posts:
        subreddit_name = post.subreddit
        post_type = 'None'

        if subreddit_name == 'coffee' or subreddit_name == 'pourover':
            post_type = coffee
        elif subreddit_name == 'tea' or subreddit_name == 'TeaPorn':
            post_type = tea

        post_id = post.name

        post_title = post.title
        post_description = post.selftext
        post_title_description = post_title + " " + post_description
        post_date = pd.to_datetime(datetime.fromtimestamp(post.created_utc).strftime("%d/%m/%Y"), format="%d/%m/%Y")

        unprocessed_tokens = tokeniser.tokenize(post_title_description)
        if unprocessed_tokens:
            if post_type == tea:
                tea_unprocessed_token_lists.append(unprocessed_tokens)
            else:
                coffee_unprocessed_token_lists.append(unprocessed_tokens)

        processed_tokens = process(post_title_description, True)
        # text, tokeniser, stop_words
        if processed_tokens:
            if post_type == tea:
                tea_processed_token_lists.append(processed_tokens)
            else:
                coffee_processed_token_lists.append(processed_tokens)

        if post.author is None:
            post_author = 'None'
        else:
            post_author = post.author.name

        if post_type == tea:
            tea_reply_graph = method.update_reply_graph_node(tea_reply_graph, post_author)
        else:
            coffee_reply_graph = method.update_reply_graph_node(coffee_reply_graph, post_author)

        # Add the post id and the author to dict of posts-ids
        post_comment_ids[post_id] = {post_id: post_author}

        # post.comments.replace_more(limit=None)
        for comment in post.comments:
            if isinstance(comment, MoreComments):
                continue

            comment_text = comment.body if comment.body is None else ''

            unprocessed_comment_tokens = tokeniser.tokenize(comment_text)
            unprocessed_tokens = unprocessed_tokens + unprocessed_comment_tokens

            if unprocessed_comment_tokens:
                if post_type == tea:
                    tea_unprocessed_token_lists.append(unprocessed_comment_tokens)
                else:
                    coffee_unprocessed_token_lists.append(unprocessed_comment_tokens)

            processed_comment_tokens = process(comment_text, False)
            processed_tokens = processed_tokens + processed_comment_tokens

            if processed_comment_tokens:
                if post_type == tea:
                    tea_processed_token_lists.append(processed_comment_tokens)
                else:
                    coffee_processed_token_lists.append(processed_comment_tokens)

            # Check if comment author exists
            comment_name = comment.name
            comment_author = comment.author
            if comment_author is not None and comment_author.name != 'ExternalUserError':
                comment_author_name = comment_author.name

                # Link the comment and comment author to the post id
                post_comment_ids[post_id].update({comment_name: comment_author_name})

                # Check whether parent comment is in the ids list
                # If not, then parent comment has been deleted
                comment_parent_id = comment.parent_id
                if comment_parent_id in post_comment_ids[post_id]:
                    if post_type == tea:

                        tea_reply_graph = method.update_reply_graph_edge(tea_reply_graph, comment_author_name,
                                                                         post_comment_ids,
                                                                         post_id, comment_parent_id)
                    else:
                        coffee_reply_graph = method.update_reply_graph_edge(coffee_reply_graph, comment_author_name,
                                                                            post_comment_ids,
                                                                            post_id, comment_parent_id)

        posts_df.loc[len(posts_df.index)] = ['reddit', post_type, post_title, post.created_utc, post_date,
                                             post_description,
                                             post_author, post.upvote_ratio, post.num_comments, unprocessed_tokens,
                                             processed_tokens]

    # Save reply graph
    nx.readwrite.write_graphml(tea_reply_graph, tea_reply_graph_filepath)
    nx.readwrite.write_graphml(coffee_reply_graph, coffee_reply_graph_filepath)
    # Read old data file if it exists to append new data collected, if not save new file

    if os.path.isfile(data_sample_filepath):
        old_posts_df = pd.read_csv(data_sample_filepath, header=0)
        posts_df = pd.concat([old_posts_df, posts_df], ignore_index=True)

    posts_df.to_csv(data_sample_filepath, index=False, header=True)

    return (tea_unprocessed_token_lists, coffee_unprocessed_token_lists, tea_processed_token_lists,
            coffee_processed_token_lists, posts_df)
