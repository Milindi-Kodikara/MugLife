# utility functions

# File containing utility functions
import re

import networkx as nx


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


def print_ego_graph(data_folder_path, ego_graph, ego_name, beverage_type):
    """
        Printing out the in and out degrees of the ego

        @param data_folder_path: folder to save the graph file
        @param ego_graph: The current user ego graph
        @param ego_name: Name of the current user we are exploring
        @param beverage_type: 'tea' or 'coffee'
    """
    # graph file name, rename to appropriate filename
    graph_filepath = f'{data_folder_path}/{beverage_type}_ego_{ego_name}.graphml'

    in_degree = ego_graph.in_degree(ego_name)
    out_degree = ego_graph.out_degree(ego_name)

    print(yellow_rgb + f'\n\nEgo name:\n{ego_name}', end='')
    print(green_rgb + f'\nIn degree of ego:\n{in_degree}', end='')
    print(red_rgb + f'\nOut degree of ego:\n{out_degree}', end='')

    in_neighbours_list = [neighbour for neighbour in ego_graph.predecessors(ego_name)]
    out_neighbours_list = [neighbour for neighbour in ego_graph.successors(ego_name)]

    # Display in and out neighbour lists
    print(green_rgb + '\nIn neighbours of ego:\n{', end='')
    print(*in_neighbours_list, sep=', ', end='')
    print('}')

    print(red_rgb + '\nOut neighbours of ego:\n{', end='')
    print(*out_neighbours_list, sep=', ', end='')
    print('}')

    # save graph
    with open(graph_filepath, 'wb') as fOut:
        nx.write_graphml(ego_graph, fOut)


def dict_to_set_format(community_dict, max_num_communities):
    """
    Converts dictionary based community output (node -> community id) to list of sets (communities)
    @param community_dict: dictionary based community representation
    @param max_num_communities: maximum number of communities

    @return: list of communities representation
    """

    # initialise
    community_list = [set() for x in range(max_num_communities)]
    # convert each (node : community id) pair to the required set format
    for (name, clus_id) in community_dict.items():
        community_list[clus_id].add(name)

    return community_list


# Graph types


type_reply = 0
type_centrality = 1
type_community = 2


def load_graphs(data_folder_path, social_media_id, graph_bev_type, graph_type_index):
    reply_graph_filepath = f'{data_folder_path}/{social_media_id}_{graph_bev_type}_reply_graph.graphml'
    centrality_graph_filepath = f'{data_folder_path}/{social_media_id}_{graph_bev_type}_modified_centrality_reply_graph.graphml'

    community_graph_filepath = f'{data_folder_path}/{social_media_id}_{graph_bev_type}_modified_community_reply_graph.graphml'

    if graph_type_index == type_centrality:
        return nx.readwrite.read_graphml(centrality_graph_filepath)
    elif graph_type_index == type_community:
        return nx.readwrite.read_graphml(community_graph_filepath)
    else:
        return nx.readwrite.read_graphml(reply_graph_filepath)
