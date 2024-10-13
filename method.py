import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import networkx as nx

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


def construct_ego_graph(client, ego, ego_name):
    """
        Constructing the ego graph for top users.

        @param client: Connection to the social media API
        @param ego: The current user instance
        @param ego_name: User name of the current user

        @returns: The constructed ego graph
    """
    ego_graph = nx.DiGraph()

    ego_graph.add_node(ego_name)

    # get all the users that have replied to a submission of the ego
    ego_submissions = ego.submissions

    for submission in ego_submissions.top(time_filter="all"):
        # get comments/replies to submission
        for replies in submission.comments:
            if replies.author != None:
                # author of comment
                replier_name = replies.author.name
                try:
                    replier_karma = replies.author.comment_karma
                except AttributeError:
                    replier_karma = 0

                # Adding nodes and edges to add follower to graph
                ego_graph.add_node(replier_name, karma=replier_karma)
                ego_graph.add_edge(replier_name, ego_name)

    # get all users that the ego has replied to
    ego_comments = ego.comments
    # get comments
    for comment in ego_comments.top(time_filter="all"):
        # get the user_id of the post that the ego's comment is replying to
        parent_comment_id = comment.parent_id
        parent_comment = client.comment(parent_comment_id)

        if parent_comment.author != None:
            replied_to_name = parent_comment.author.name
            try:
                replied_karma = parent_comment.author.comment_karma
            except(AttributeError):
                replied_karma = 0

            # Add nodes and edges to add follower to graph
            ego_graph.add_node(replied_to_name, karma=replied_karma)
            ego_graph.add_edge(ego_name, replied_to_name)

    return ego_graph


def update_reply_graph_node(reply_graph, post_author):
    """
        Updating the reply graph upon encountering a new post.

        @param reply_graph: Reply graph instance
        @param post_author: Author of the post

        @returns: The updated reply graph
    """
    # Check if author is in the reply graph
    # If author is in, update the no. of posts
    # Else, create a new node for the author
    if post_author in reply_graph:
        reply_graph.nodes[post_author]['subNum'] += 1
    else:
        reply_graph.add_node(post_author, subNum=1)

    return reply_graph


def update_reply_graph_edge(reply_graph, comment_author_name, post_comment_ids, post_id, comment_parent_id):
    """
        Updating the reply graph upon encountering a new comment.

        @param reply_graph: Reply graph instance
        @param comment_author_name: Author of the comment
        @param post_comment_ids: List of post, post id, associated comments and comment ids
        @param post_id: Current post id
        @param comment_parent_id: Parent comment id of the current comment

        @returns: The updated reply graph
    """

    # If edge exists, increment the replyNum,
    # else, add a new edge

    if reply_graph.has_edge(comment_author_name, post_comment_ids[post_id][comment_parent_id]):
        reply_graph[comment_author_name][post_comment_ids[post_id][comment_parent_id]]['replyNum'] += 1
    else:
        # need to check if the nodes have been added yet, if not add it and set subNum to 0
        if comment_author_name not in reply_graph:
            reply_graph.add_node(comment_author_name, subNum=0)

        if not post_comment_ids[post_id][comment_parent_id] in reply_graph:
            reply_graph.add_node(post_comment_ids[post_id][comment_parent_id], subNum=0)

        reply_graph.add_edge(comment_author_name, post_comment_ids[post_id][comment_parent_id], replyNum=1)

    return reply_graph
