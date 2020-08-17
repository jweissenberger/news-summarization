import tensorflow as tf
import numpy as np
import logging
from typing import List
import sentencepiece as sentencepiece_processor


def create_text_encoder(encoder_type: str, vocab_filename: str):
  if encoder_type == "sentencepiece":
    return SentencePieceEncoder(vocab_filename)
  elif encoder_type == "sentencepiece_newline":
    return SentencePieceEncoder(vocab_filename, newline_symbol="<n>")
  else:
    raise ValueError("Unsupported encoder type: %s" % encoder_type)


class SentencePieceEncoder(object):
  """SentencePieceEncoder.

  First two ids are pad=0, eos=1, rest ids are being shifted up by
  shift_reserved_tokens. If newline_symbol is provided, will replace newline in
  the text with that token.
  """

  def __init__(self,
               sentencepiece_model_file: str,
               shift_reserved_tokens: int = 103,
               newline_symbol: str = ""):
    self._tokenizer = sentencepiece_processor.SentencePieceProcessor()
    self._sp_model = tf.io.gfile.GFile(sentencepiece_model_file, "rb").read()
    self._tokenizer.LoadFromSerializedProto(self._sp_model)
    self._shift_reserved_tokens = shift_reserved_tokens
    self._newline_symbol = newline_symbol

  @property
  def vocab_size(self) -> int:
    return self._tokenizer.GetPieceSize() + self._shift_reserved_tokens

  def encode(self, text: str) -> List[int]:
    if self._newline_symbol:
      text = text.replace("\n", self._newline_symbol)
    ids = self._tokenizer.EncodeAsIds(text)
    ids = [i + self._shift_reserved_tokens if i > 1 else i for i in ids]
    return ids

  def decode(self, ids: List[int]) -> str:
    ids = [
        i - self._shift_reserved_tokens
        if i > 1 + self._shift_reserved_tokens else i for i in ids
    ]
    text = self._tokenizer.DecodeIds(ids)
    if self._newline_symbol:
      text = text.replace(self._newline_symbol, "\n")
    return text



def ids2str(encoder, ids, num_reserved):
  """Decode ids."""
  if num_reserved:
    eos = np.where(ids == 1)[0]
    if np.any(eos):
      ids = ids[:eos[0]]
    reserved_tokens = np.where(ids < num_reserved)[0]
    if reserved_tokens.size:
      split_locations = np.union1d(reserved_tokens, reserved_tokens + 1)
      ids_list = np.split(ids, split_locations)
      text_list = [
          "<%d>" %
          i if len(i) == 1 and i < num_reserved else encoder.decode(i.tolist())
          for i in ids_list
      ]
      return " ".join(text_list)
  return encoder.decode(ids.flatten().tolist())


def run_summarization(text, model_name='cnn_dailymail', model_dir='model/cnn_dailymail/'):

    _SPM_VOCAB = 'ckpt/c4.unigram.newline.10pct.96000.model'
    encoder = create_text_encoder("sentencepiece", _SPM_VOCAB)
    shapes = {
        'cnn_dailymail': (1024, 128),
        'gigaword': (128, 32)
    }

    tf.get_logger().setLevel(logging.ERROR)

    shape,_ = shapes[model_name]

    input_ids = encoder.encode(text)
    inputs = np.zeros(shape)
    idx = len(input_ids)
    if idx>shape: idx =shape

    inputs[:idx] = input_ids[:idx]
    imported = tf.saved_model.load(model_dir, tags='serve')
    example = tf.train.Example()
    example.features.feature["inputs"].int64_list.value.extend(inputs.astype(int))
    output = imported.signatures['serving_default'](examples=tf.constant([example.SerializeToString()]))
    prediction = ids2str(encoder, output['outputs'].numpy(), None)

    return prediction


def part_by_part_summarization(text, model_name='cnn_dailymail', model_dir='model/cnn_dailymail/'):
    """

    :param text:
    :param part_size: Number of sentences you want to summarize at a time
    :param model_name:
    :param model_dir:
    :return:
    """


    # TODO this needs to be a split on every 5 sentences or so. (use NLTK sentence tokenizer)
    # split on paragraphs
    paragraphs = text.split('\n')

    summary = ""

    chunk_to_summarize = ""
    for paragraph in paragraphs:
        chunk_to_summarize += paragraph

        # if there are less than 100 words in the paragraph also grab the next paragraph
        if len(chunk_to_summarize.split(' ')) < 75:
            continue
        else:
            summary += '. ' + run_summarization(text=chunk_to_summarize, model_name=model_name, model_dir=model_dir)
            chunk_to_summarize = ""

    return summary


if __name__ == '__main__':

    file = open("cnn.txt", "r")
    article = file.read()
    file.close()

    print(part_by_part_summarization(article, model_name='gigaword', model_dir='model/gigaword/'))
    #print(run_summarization(article))
