from nltk.tokenize import sent_tokenize


def sentence_tokenizer(text):
    """
    Sentence tokenizer
    :param text: string of text to be separated by sentences
    :return: list of sentences
    """
    text = text.replace('*', '')
    text = text.replace('#', '')

    # keep quotes together
    # (previously it would break up sentences within quotes which would create choppy summaries, this prevents that)
    in_quote = False
    new_text = ''
    for char in text:
        if char == '"':
            in_quote = not in_quote

        if in_quote and char == '.':
            new_text += '<quote_period>'
        else:
            new_text += char
    text = new_text

    sentences = sent_tokenize(text)

    output = []
    for sentence in sentences:
        # remove super short sentences (usually titles or numbers or something)
        if len(sentence) < 8:
            continue
        else:
            output.append(sentence)

    for i in range(len(output)):
        output[i] = output[i].replace('<quote_period>', '.')

    return output


def capitalization_fix(text):
    # Thought I might need this to clean up the output from the models but don't think I need it
    # look at the pip package truecase if this is needed in the future
    raise NotImplementedError()


def plagiarism_checker(new_text, orig_text):

    splits = new_text.split(' ')

    new_plagiarism = {}  # key is the number of the word in orig text and value is binary for if plagiarized
    for i in range(len(splits)):
        new_plagiarism[i] = False

    len_chunk = 3
    for i in range(len(splits) + 1 - len_chunk):
        chunk = splits[i:i + len_chunk]

        chunk_text = ' '.join(chunk)

        if chunk_text in orig_text:
            for j in range(i, i + len_chunk):
                new_plagiarism[j] = True

    open_char = '<span style="color:red;">'
    close = '</span>'
    output = ''
    words_plagiarized = 0
    for i in range(len(splits)):
        if new_plagiarism[i]:
            words_plagiarized += 1

        # first element is stolen
        if i == 0 and new_plagiarism[0]:
            output += open_char + splits[i] + ' '
            continue

        # non first element is stolen (first of bunch)
        if new_plagiarism[i] and not new_plagiarism[i - 1]:
            output += open_char + splits[i] + ' '
            continue

        # last element is stolen
        if new_plagiarism[i] and i == len(splits) - 1:
            output += splits[i] + close
            continue

        # middle of a bunch of stolen elements
        if new_plagiarism[i] and new_plagiarism[i+1] and new_plagiarism[i - 1]:
            output += splits[i] + ' '

        # end of a bunch of stolen
        if new_plagiarism[i] and not new_plagiarism[i + 1]:
            output += splits[i] + close + ' '
            continue

        # element not stolen
        if not new_plagiarism[i]:
            output += splits[i] + ' '

    # calculate % plagiarism
    percent_plagiarism = (words_plagiarized / len(splits)) * 100
    output = f'Percent Plagiarism: {percent_plagiarism}%<br>' + output

    return output


def new_text_checker(new_text, orig_text):
    """
    Given new text and old text, puts any new words (those not directly quoted in brackets)

    ex:
    orig_text = 'He said, "Build the wall"'
    new_text = 'Trump said, "Build the wall"'

    would return -> '[Trump] said, "Build the wall"'

    :param new_text:
    :param orig_text:
    :return:
    """

    splits = new_text.split(' ')

    new_plagiarism = {}  # key is the number of the word in orig text and value is binary for if plagiarized
    for i in range(len(splits)):
        new_plagiarism[i] = False

    len_chunk = 3
    for i in range(len(splits) + 1 - len_chunk):
        chunk = splits[i:i+len_chunk]

        chunk_text = ' '.join(chunk)

        if chunk_text in orig_text:
            for j in range(i, i+len_chunk):
                new_plagiarism[j] = True

    output = ''
    in_bracket = False
    for i in range(len(splits)):

        # case not plagiarized
        if not new_plagiarism[i]:
            if i == 0:
                output += '[' + splits[i]
                in_bracket = True
                continue

            if i == len(splits) - 1:
                if not in_bracket:
                    output += ' ['
                else:
                    output += ' '
                output += splits[i] + ']'
                break

            if not in_bracket:
                in_bracket = True
                output += f" [{splits[i]}"
                continue

            if in_bracket:
                output += f" {splits[i]}"

        else:
            if in_bracket:
                output += f'] {splits[i]}'
                in_bracket = False
            else:
                output += f' {splits[i]}'

    return output


def clean_text(text):

    text = text.replace('&', 'and')

    allowed_symbols = ['"', "'", ' ', '$', ':', '.', '?', '!', '(', ')', '/', ';']
    allowed_symbols = set(allowed_symbols)  # allow for quicker check even though this it will only save like 5 operations but would scale better

    new_text = ""
    for char in text:
        if char.isalnum() or char in allowed_symbols:
            new_text += char

    return new_text
