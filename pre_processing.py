# helper functions for pre-processing
import re
import utils


def process(text, tokeniser, stemmer, stop_words, print_processing=False):
    """
        Perform the processing of the reddit posts
        @param text: the text (reddit post/comment) to process
        @param tokeniser
        @param stemmer
        @param stop_words: list of stop words to remove from text
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
