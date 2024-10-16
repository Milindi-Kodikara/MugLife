# helper functions for visualising data
import networkx as nx

import method
import utils
from collections import Counter
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["figure.autolayout"] = True

sns.set(style='whitegrid', palette='Dark2')


def generic_chart(title, x_label, y_label):
    """
        Create a chart using matplotlib

        @param title: Title of the chart
        @param x_label: Label of the x-axis
        @param y_label: Label of the y-axis
    """
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.show()


def generate_bar_chart(x, y, color, title, x_label, y_label):
    """
        Create a bar chart using matplotlib

        @param x: List of x-axis tick values
        @param y: List of y-axis tick values
        @param color: Bar colour
        @param title: Title of the bar chart
        @param x_label: Label of the x-axis
        @param y_label: Label for the y-axis
    """
    plt.bar(x, y, color=color)
    generic_chart(title, x_label, y_label)


def compute_term_freq(beverage_type, token_list, generate_visual, color=utils.green):
    """
        Calculate the term frequency of the corpus
        @param beverage_type: 'Tea' or 'Coffee'
        @param token_list: list of processed tokens
        @param generate_visual: Bool to determine if the visual should be created
        @param color: bar colour of the bars of the bar graph

        Generates a visual representation (bar graph) of the term frequency
    """
    term_freq = 50
    term_freq_counter = Counter()

    term_freq_counter.update(token_list)

    print("-----------------\nTerm frequency\n-----------------\n")
    for term, count in term_freq_counter.most_common(term_freq):
        print(term + ': ' + str(count))
    print("----------------------------------\n")

    if generate_visual:
        y = [count for term, count in term_freq_counter.most_common(term_freq)]
        x = [term for term, count in term_freq_counter.most_common(term_freq)]

        generate_bar_chart(x, y, color, f"Term frequency distribution for {beverage_type}", 'Term frequency',
                           'Number of words with term frequency')


def generate_time_series(item_list, title, x_column, y_column, x_label, y_label, color):
    """
        Create a timeseries using matplotlib

        @param item_list: List of items to be plotted, eg: [[Date, Value],...]
        @param title: Title of the plot
        @param x_column: Name of the column to be converted and used for x-axis
        @param y_column: Name of the column to be used for y-axis
        @param x_label: Label of the x-axis
        @param y_label: Label for the y-axis
        @param color: Line colour
    """
    series = pd.DataFrame(item_list, columns=[x_column, y_column])
    series.set_index(x_column, inplace=True)
    series[[y_column]] = series[[y_column]].apply(pd.to_numeric)
    new_series = series.resample('1D').sum()
    new_series.plot(color=color)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def display_time_series_stats(series, function, title, x_label, y_label, color):
    """
        Create a timeseries using matplotlib and print out the required information

        @param series: The series object to be plotted, eg: Date against Reddit post count
        @param function: The analysis function (eg: sum, count) used
        @param title: The title of the plot
        @param x_label: The label for the x-axis
        @param y_label: The label for the y-axis
        @param color: Colour of the line of the plot
    """
    ordered = series.reset_index(name=function).sort_values([function], ascending=False)
    print(f'{title} ordered:\n{ordered.head()}')

    df = pd.DataFrame(columns=['Date', 'Values'])
    df['Date'] = series.index
    df['Values'] = series.to_list()

    x_column = 'Date'
    y_column = 'Values'

    combined_list = [[row.Date, row.Values] for row in df.itertuples()]
    generate_time_series(combined_list, title, x_column, y_column, x_label, y_label, color)


def display_topics(model, feature_names, num_top_words):
    """
        Print out the most associated words for each feature.

        @param model: LDA model
        @param feature_names: list of strings, representing the list of features/words.
        @param num_top_words: number of words to print per topic.
    """
    # print out the topic distributions
    for topic_id, topic_distribution_list in enumerate(model.components_):
        print("Topic %d:" % (topic_id))
        print(" ".join([feature_names[i] for i in topic_distribution_list.argsort()[:-num_top_words - 1:-1]]))


def display_word_cloud(model, feature_names):
    """
    Displays the word cloud of the topic distributions, stored in model.

    @param model: LDA model
    @param feature_names: list of strings, representing the list of features/words
    """
    # this normalises each row/topic to sum to one
    # normalised_components to display word clouds
    normalised_components = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]

    topic_num = len(model.components_)
    # number of wordclouds for each row
    plot_col_num = 2
    # number of wordclouds for each column
    plot_row_num = int(math.ceil(topic_num / plot_col_num))

    for topicId, lTopicDist in enumerate(normalised_components):
        l_word_prob = {feature_names[i]: wordProb for i, wordProb in enumerate(lTopicDist)}
        wordcloud = WordCloud(background_color='black')
        wordcloud.fit_words(frequencies=l_word_prob)
        plt.subplot(plot_row_num, plot_col_num, topicId + 1)
        plt.title('Topic %d:' % (topicId + 1))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

    plt.show(block=True)


def display_networkx_graph(graph, title):
    """
        Display a networkx graph for the given user

        @param graph: The current networkx graph
        @param title: title for the graph
    """
    plt.figure(figsize=(20, 20))
    nx.draw_networkx(graph, arrows=True, with_labels=True, pos=nx.kamada_kawai_layout(graph, scale=10))
    plt.axis('off')
    plt.title(title)
    plt.show()


def display_centrality_histograms(degree_centrality_list, eigen_vector_centrality_list, katz_centrality_list, color):
    """
        Display histograms for centrality

        @param degree_centrality_list: Centrality list
        @param eigen_vector_centrality_list: Eigen vector centrality list
        @param katz_centrality_list: Katz centrality list
        @param color: Bar color
    """

    plt.subplot(1, 3, 1)
    plt.hist(list(degree_centrality_list.values()), color=color)
    plt.title('Degree')
    plt.xlabel('Centrality')

    plt.subplot(1, 3, 2)
    plt.hist(list(eigen_vector_centrality_list.values()), color=color)
    plt.title('Eigenvector')
    plt.xlabel('Centrality')

    plt.subplot(1, 3, 3)
    plt.hist(list(katz_centrality_list.values()), color=color)
    plt.title('Katz')
    plt.xlabel('Centrality')

    plt.show()


# TODO: Add the desc here
def display_linear_threshold_stats(trial_num, list_of_seeds, graph_to_explore, prefix_filepath):
    if graph_to_explore != None:
        average_activations_per_node_list, average_activations_per_iteration_list = method.compute_linear_threshold(
            graph_to_explore,
            trial_num, list_of_seeds)

        print(f'Average activations per node:\n{average_activations_per_node_list}')
        print(f'Average activations per iteration:\n{average_activations_per_iteration_list}')

        total_nodes = nx.number_of_nodes(graph_to_explore)
        print('\n------------Linear threshold graph exploration------------\n')
        if len(average_activations_per_iteration_list) > 0:

            average_nodes_activated = sum(average_activations_per_iteration_list) / len(
                average_activations_per_iteration_list)

            print(utils.green_rgb, f'Average number of nodes activated: {average_nodes_activated} out of {total_nodes}',
                  end='')
        else:
            print(utils.green_rgb, f'Average number of nodes activated: 0 out of {total_nodes}',
                  end='')

        # Save to graph
        # average activation per node for the cascade graph,
        # stored in node attribute 'avgAct'
        for node_id, avg_activation in enumerate(average_activations_per_node_list):
            graph_to_explore.nodes[node_id]['avgAct'] = avg_activation

        # Output modified graphs to respective files
        linear_threshold_graph_filepath = f'{prefix_filepath}_linear_threshold_graph.graphml'

        nx.readwrite.write_graphml(graph_to_explore, linear_threshold_graph_filepath, infer_numeric_types=True)


def display_tree_graph(trial_num, list_of_seeds, graph_to_display, prefix_filepath):
    branching_factor = 2
    tree_height = 5

    tree_graph = nx.balanced_tree(r=branching_factor, h=tree_height, create_using=graph_to_display)

    tree_graph = utils.generate_weights(tree_graph)

    prefix_filepath = f'{prefix_filepath}_tree'
    display_linear_threshold_stats(trial_num, list_of_seeds, tree_graph, prefix_filepath)

    nx.draw_networkx(tree_graph, arrows=False, with_labels=True)


def display_barabasi_albert_graph(trial_num, list_of_seeds, graph_to_display, prefix_filepath):
    num_nodes = graph_to_display.number_of_nodes()
    num_edges = graph_to_display.number_of_edges()

    small_world_graph = nx.barabasi_albert_graph(n=num_nodes, m=num_edges)

    small_world_graph = utils.generate_weights(small_world_graph)

    prefix_filepath = f'{prefix_filepath}_small_world'
    display_linear_threshold_stats(trial_num, list_of_seeds, small_world_graph, prefix_filepath)

    nx.draw_networkx(small_world_graph, arrows=True, with_labels=True)
