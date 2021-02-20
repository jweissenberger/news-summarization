from textblob import TextBlob

from common import sentence_tokenizer


def textblob_subjectivity_analysis(text):
    """
    The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.

    Seems to do better on neutral sentences

    :param text:
    :return:
    subjectivity: float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective
    """

    return TextBlob(text).sentiment.subjectivity


def textblob_sentence_by_sentence_subjectivity(text):
    sentences = sentence_tokenizer(text)

    output = []
    for sent in sentences:
        subjectivity = textblob_subjectivity_analysis(sent)

        output.append((subjectivity, sent))

    output = sorted(output, key=lambda x: x[0])

    return output


def textblob_topn_subjectivity(text, num_sentences=3):

    sentence_subjectivity = textblob_sentence_by_sentence_subjectivity(text)

    most_subjective = sentence_subjectivity[-num_sentences:]
    least_subjective = sentence_subjectivity[:num_sentences]

    return most_subjective, least_subjective


def run_subjectivity(text, num_sentences=3):

    print("******************Full Subjectivity Analysis******************\n\n")

    most_subjective, least_subjective = textblob_topn_subjectivity(text, num_sentences=num_sentences)

    print("Most Subjective:\n")
    for i in reversed(most_subjective):
        print("Score:", i[0])
        print(f"Sentence:\n{i[1]}\n")

    print("\n\n*********")
    print("Least Subjective:")
    for i in least_subjective:
        print("Score:", i[0])
        print(f"Sentence:\n{i[1]}\n")


if __name__ == '__main__':


    file = open("fox.txt", "r")
    fox = file.read()
    file.close()

    run_subjectivity(fox, num_sentences=3)
