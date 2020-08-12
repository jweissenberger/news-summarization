import public_parsing_ops
import tensorflow as tf
import numpy as np
import logging
from typing import List
import sentencepiece as sentencepiece_processor


# python test_example.py --article cnn.txt --model_dir model/cnn_dailymail/ --model_name cnn_dailymail
# python test_example.py --article cnn.txt --model_dir model/gigaword/ --model_name gigaword

_SPM_VOCAB = 'ckpt/c4.unigram.newline.10pct.96000.model'
encoder = public_parsing_ops.create_text_encoder("sentencepiece",
                                                 _SPM_VOCAB)
shapes = {
    'cnn_dailymail': (1024, 128),
    'gigaword':(128, 32)
}

_SHIFT_RESERVED_TOKENS = 103
_NEWLINE_SYMBOL = "<n>"


def create_text_encoder(encoder_type: str, vocab_filename: str):
  if encoder_type == "sentencepiece":
    return SentencePieceEncoder(vocab_filename)
  elif encoder_type == "sentencepiece_newline":
    return SentencePieceEncoder(vocab_filename, newline_symbol=_NEWLINE_SYMBOL)
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
               shift_reserved_tokens: int = _SHIFT_RESERVED_TOKENS,
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


if __name__ == '__main__':
    import argparse

    tf.get_logger().setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", help="path of your example article", default="example_article")
    parser.add_argument("--model_dir", help="path of your model directory", default="model/")
    parser.add_argument("--model_name", help="path of your model directory", default="cnn_dailymail")
    args = parser.parse_args()

    text = "Partisanship at every turn: On the same day a new Gallup poll came out showing an 84-point gap between Republican and Democratic approval of Trump, the bitter divide in Congress -- and the country -- was visible everywhere during the President's speech." #open(args.article, "r", encoding="utf-8").read()

    shape,_ = shapes[args.model_name]

    input_ids = encoder.encode(text)
    inputs = np.zeros(shape)
    idx = len(input_ids)
    if idx>shape: idx =shape

    inputs[:idx] = input_ids[:idx]
    imported = tf.saved_model.load(args.model_dir, tags='serve')
    example = tf.train.Example()
    example.features.feature["inputs"].int64_list.value.extend(inputs.astype(int))
    output = imported.signatures['serving_default'](examples=tf.constant([example.SerializeToString()]))

    print("\nPREDICTION >> ", ids2str(encoder, output['outputs'].numpy(), None))