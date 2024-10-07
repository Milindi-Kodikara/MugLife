# helper functions for visualising data
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


def compute_term_freq(token_list, generate_visual, color=utils.green):
    """
        Calculate the term frequency of the corpus
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

        generate_bar_chart(x, y, color, "Term frequency distribution", 'Term frequency',
                           'Number of words with term frequency')
