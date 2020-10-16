# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Config controlling hyperparameters for fine-tuning ELECTRA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v1 as tf


class FinetuningConfig(object):
  """Fine-tuning hyperparameters."""

  def __init__(self, model_name, data_dir, **kwargs):
    # general
    self.model_name = model_name
    self.debug = False  # debug mode for quickly running things
    self.log_examples = False  # print out some train examples for debugging
    self.num_trials = 1  # how many train+eval runs to perform
    self.do_train = True  # train a model
    self.do_eval = True  # evaluate the model
    self.keep_all_models = True  # if False, only keep the last trial's ckpt

    # model
    self.model_size = "small"  # one of "small", "base", or "large"
    self.task_names = ["chunk"]  # which tasks to learn
    # override the default transformer hparams for the provided model size; see
    # modeling.BertConfig for the possible hparams and util.training_utils for
    # the defaults
    self.model_hparam_overrides = (
        kwargs["model_hparam_overrides"]
        if "model_hparam_overrides" in kwargs else {})
    self.embedding_size = None  # bert hidden size by default
    self.vocab_size = 30522  # number of tokens in the vocabulary
    self.do_lower_case = True

    # training
    self.learning_rate = 1e-4
    self.weight_decay_rate = 0.01
    self.layerwise_lr_decay = 0.8  # if > 0, the learning rate for a layer is
                                   # lr * lr_decay^(depth - max_depth) i.e.,
                                   # shallower layers have lower learning rates
    self.num_train_epochs = 3.0  # passes over the dataset during training
    self.warmup_proportion = 0.1  # how much of training to warm up the LR for
    self.save_checkpoints_steps = 1000000
    self.iterations_per_loop = 1000
    self.use_tfrecords_if_existing = True  # don't make tfrecords and write them
                                           # to disc if existing ones are found

    # writing model outputs to disc
    self.write_test_outputs = False  # whether to write test set outputs,
                                     # currently supported for GLUE + SQuAD 2.0
    self.n_writes_test = 5  # write test set predictions for the first n trials

    # sizing
    self.max_seq_length = 128
    self.train_batch_size = 32
    self.eval_batch_size = 32
    self.predict_batch_size = 32
    self.double_unordered = True  # for tasks like paraphrase where sentence
                                  # order doesn't matter, train the model on
                                  # on both sentence orderings for each example
    # for qa tasks
    self.max_query_length = 64   # max tokens in q as opposed to context
    self.doc_stride = 128  # stride when splitting doc into multiple examples
    self.n_best_size = 20  # number of predictions per example to save
    self.max_answer_length = 30  # filter out answers longer than this length
    self.answerable_classifier = True  # answerable classifier for SQuAD 2.0
    self.answerable_uses_start_logits = True  # more advanced answerable
                                              # classifier using predicted start
    self.answerable_weight = 0.5  # weight for answerability loss
    self.joint_prediction = True  # jointly predict the start and end positions
                                  # of the answer span
    self.beam_size = 20  # beam size when doing joint predictions
    self.qa_na_threshold = -2.75  # threshold for "no answer" when writing SQuAD
                                  # 2.0 test outputs

    # TPU settings
    self.use_tpu = False
    self.num_tpu_cores = 1
    self.tpu_job_name = None
    self.tpu_name = None  # cloud TPU to use for training
    self.tpu_zone = None  # GCE zone where the Cloud TPU is located in
    self.gcp_project = None  # project name for the Cloud TPU-enabled project

    # default locations of data files
    self.data_dir = data_dir
    pretrained_model_dir = os.path.join(data_dir, "models", model_name)
    self.raw_data_dir = os.path.join(data_dir, "finetuning_data", "{:}").format
    self.vocab_file = os.path.join(pretrained_model_dir, "vocab.txt")
    if not tf.io.gfile.exists(self.vocab_file):
      self.vocab_file = os.path.join(self.data_dir, "vocab.txt")
    task_names_str = ",".join(
        kwargs["task_names"] if "task_names" in kwargs else self.task_names)
    self.init_checkpoint = None if self.debug else pretrained_model_dir
    self.model_dir = os.path.join(pretrained_model_dir, "finetuning_models",
                                  task_names_str + "_model")
    results_dir = os.path.join(pretrained_model_dir, "results")
    self.results_txt = os.path.join(results_dir,
                                    task_names_str + "_results.txt")
    self.results_pkl = os.path.join(results_dir,
                                    task_names_str + "_results.pkl")
    qa_topdir = os.path.join(results_dir, task_names_str + "_qa")
    self.qa_eval_file = os.path.join(qa_topdir, "{:}_eval.json").format
    self.qa_preds_file = os.path.join(qa_topdir, "{:}_preds.json").format
    self.qa_na_file = os.path.join(qa_topdir, "{:}_null_odds.json").format
    self.preprocessed_data_dir = os.path.join(
        pretrained_model_dir, "finetuning_tfrecords",
        task_names_str + "_tfrecords" + ("-debug" if self.debug else ""))
    self.test_predictions = os.path.join(
        pretrained_model_dir, "test_predictions",
        "{:}_{:}_{:}_predictions.pkl").format

    # update defaults with passed-in hyperparameters
    self.update(kwargs)

    # default hyperparameters for single-task models
    if len(self.task_names) == 1:
      task_name = self.task_names[0]
      if task_name == "rte" or task_name == "sts":
        self.num_train_epochs = 10.0
      elif "squad" in task_name or "qa" in task_name:
        self.max_seq_length = 512
        self.num_train_epochs = 2.0
        self.write_distill_outputs = False
        self.write_test_outputs = False
      elif task_name == "chunk":
        self.max_seq_length = 256
      else:
        self.num_train_epochs = 3.0

    # default hyperparameters for different model sizes
    if self.model_size == "large":
      self.learning_rate = 5e-5
      self.layerwise_lr_decay = 0.9
    elif self.model_size == "small":
      self.embedding_size = 128

    # debug-mode settings
    if self.debug:
      self.save_checkpoints_steps = 1000000
      self.use_tfrecords_if_existing = False
      self.num_trials = 1
      self.iterations_per_loop = 1
      self.train_batch_size = 32
      self.num_train_epochs = 3.0
      self.log_examples = True

    # passed-in-arguments override (for example) debug-mode defaults
    self.update(kwargs)

  def update(self, kwargs):
    for k, v in kwargs.items():
      if k not in self.__dict__:
        raise ValueError("Unknown hparam " + k)
      self.__dict__[k] = v

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text classification and regression tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import csv
import os
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import feature_spec
from finetune import task
from finetune.classification import classification_metrics
from model import tokenization
from util import utils


class InputExample(task.Example):
  """A single training/test example for simple sequence classification."""

  def __init__(self, eid, task_name, text_a, text_b=None, label=None):
    super(InputExample, self).__init__(task_name)
    self.eid = eid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class SingleOutputTask(task.Task):
  """Task with a single prediction per example (e.g., text classification)."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer):
    super(SingleOutputTask, self).__init__(config, name)
    self._tokenizer = tokenizer

  def get_examples(self, split):
    return self._create_examples(read_tsv(
        os.path.join(self.config.raw_data_dir(self.name), split + ".tsv"),
        max_lines=100 if self.config.debug else None), split)

  @abc.abstractmethod
  def _create_examples(self, lines, split):
    pass

  def featurize(self, example: InputExample, is_training, log=False):
    """Turn an InputExample into a dict of features."""
    tokens_a = self._tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
      tokens_b = self._tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, self.config.max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > self.config.max_seq_length - 2:
        tokens_a = tokens_a[0:(self.config.max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it
    # makes it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < self.config.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length

    if log:
      utils.log("  Example {:}".format(example.eid))
      utils.log("    tokens: {:}".format(" ".join(
          [tokenization.printable_text(x) for x in tokens])))
      utils.log("    input_ids: {:}".format(" ".join(map(str, input_ids))))
      utils.log("    input_mask: {:}".format(" ".join(map(str, input_mask))))
      utils.log("    segment_ids: {:}".format(" ".join(map(str, segment_ids))))

    eid = example.eid
    features = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_eid": eid,
    }
    self._add_features(features, example, log)
    return features

  def _load_glue(self, lines, split, text_a_loc, text_b_loc, label_loc,
                 skip_first_line=False, eid_offset=0, swap=False):
    examples = []
    for (i, line) in enumerate(lines):
      try:
        if i == 0 and skip_first_line:
          continue
        eid = i - (1 if skip_first_line else 0) + eid_offset
        text_a = tokenization.convert_to_unicode(line[text_a_loc])
        if text_b_loc is None:
          text_b = None
        else:
          text_b = tokenization.convert_to_unicode(line[text_b_loc])
        if "test" in split or "diagnostic" in split:
          label = self._get_dummy_label()
        else:
          label = tokenization.convert_to_unicode(line[label_loc])
        if swap:
          text_a, text_b = text_b, text_a
        examples.append(InputExample(eid=eid, task_name=self.name,
                                     text_a=text_a, text_b=text_b, label=label))
      except Exception as ex:
        utils.log("Error constructing example from line", i,
                  "for task", self.name + ":", ex)
        utils.log("Input causing the error:", line)
    return examples

  @abc.abstractmethod
  def _get_dummy_label(self):
    pass

  @abc.abstractmethod
  def _add_features(self, features, example, log):
    pass


class RegressionTask(SingleOutputTask):
  """Task where the output is a real-valued score for the input text."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer, min_value, max_value):
    super(RegressionTask, self).__init__(config, name, tokenizer)
    self._tokenizer = tokenizer
    self._min_value = min_value
    self._max_value = max_value

  def _get_dummy_label(self):
    return 0.0

  def get_feature_specs(self):
    feature_specs = [feature_spec.FeatureSpec(self.name + "_eid", []),
                     feature_spec.FeatureSpec(self.name + "_targets", [],
                                              is_int_feature=False)]
    return feature_specs

  def _add_features(self, features, example, log):
    label = float(example.label)
    assert self._min_value <= label <= self._max_value
    # simple normalization of the label
    label = (label - self._min_value) / self._max_value
    if log:
      utils.log("    label: {:}".format(label))
    features[example.task_name + "_targets"] = label

  def get_prediction_module(self, bert_model, features, is_training,
                            percent_done):
    reprs = bert_model.get_pooled_output()
    if is_training:
      reprs = tf.nn.dropout(reprs, keep_prob=0.9)

    predictions = tf.layers.dense(reprs, 1)
    predictions = tf.squeeze(predictions, -1)

    targets = features[self.name + "_targets"]
    losses = tf.square(predictions - targets)
    outputs = dict(
        loss=losses,
        predictions=predictions,
        targets=features[self.name + "_targets"],
        eid=features[self.name + "_eid"]
    )
    return losses, outputs

  def get_scorer(self):
    return classification_metrics.RegressionScorer()


class ClassificationTask(SingleOutputTask):
  """Task where the output is a single categorical label for the input text."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, config: configure_finetuning.FinetuningConfig, name,
               tokenizer, label_list):
    super(ClassificationTask, self).__init__(config, name, tokenizer)
    self._tokenizer = tokenizer
    self._label_list = label_list

  def _get_dummy_label(self):
    return self._label_list[0]

  def get_feature_specs(self):
    return [feature_spec.FeatureSpec(self.name + "_eid", []),
            feature_spec.FeatureSpec(self.name + "_label_ids", [])]

  def _add_features(self, features, example, log):
    label_map = {}
    for (i, label) in enumerate(self._label_list):
      label_map[label] = i
    label_id = label_map[example.label]
    if log:
      utils.log("    label: {:} (id = {:})".format(example.label, label_id))
    features[example.task_name + "_label_ids"] = label_id

  def get_prediction_module(self, bert_model, features, is_training,
                            percent_done):
    num_labels = len(self._label_list)
    reprs = bert_model.get_pooled_output()

    if is_training:
      reprs = tf.nn.dropout(reprs, keep_prob=0.9)

    logits = tf.layers.dense(reprs, num_labels)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = features[self.name + "_label_ids"]
    labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)

    losses = -tf.reduce_sum(labels * log_probs, axis=-1)

    outputs = dict(
        loss=losses,
        logits=logits,
        predictions=tf.argmax(logits, axis=-1),
        label_ids=label_ids,
        eid=features[self.name + "_eid"],
    )
    return losses, outputs

  def get_scorer(self):
    return classification_metrics.AccuracyScorer()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def read_tsv(input_file, quotechar=None, max_lines=None):
  """Reads a tab separated value file."""
  with tf.io.gfile.GFile(input_file, "r") as f:
    reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
    lines = []
    for i, line in enumerate(reader):
      if max_lines and i >= max_lines:
        break
      lines.append(line)
    return lines


class MNLI(ClassificationTask):
  """Multi-NLI."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(MNLI, self).__init__(config, "mnli", tokenizer,
                               ["contradiction", "entailment", "neutral"])

  def get_examples(self, split):
    if split == "dev":
      split += "_matched"
    return self._create_examples(read_tsv(
        os.path.join(self.config.raw_data_dir(self.name), split + ".tsv"),
        max_lines=100 if self.config.debug else None), split)

  def _create_examples(self, lines, split):
    if split == "diagnostic":
      return self._load_glue(lines, split, 1, 2, None, True)
    else:
      return self._load_glue(lines, split, 8, 9, -1, True)

  def get_test_splits(self):
    return ["test_matched", "test_mismatched", "diagnostic"]


class MRPC(ClassificationTask):
  """Microsoft Research Paraphrase Corpus."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(MRPC, self).__init__(config, "mrpc", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    examples = []
    examples += self._load_glue(lines, split, 3, 4, 0, True)
    if self.config.double_unordered and split == "train":
      examples += self._load_glue(
          lines, split, 3, 4, 0, True, len(examples), True)
    return examples


class CoLA(ClassificationTask):
  """Corpus of Linguistic Acceptability."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(CoLA, self).__init__(config, "cola", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    return self._load_glue(lines, split, 1 if split == "test" else 3,
                           None, 1, split == "test")

  def get_scorer(self):
    return classification_metrics.MCCScorer()


class SST(ClassificationTask):
  """Stanford Sentiment Treebank."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(SST, self).__init__(config, "sst", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    if "test" in split:
      return self._load_glue(lines, split, 1, None, None, True)
    else:
      return self._load_glue(lines, split, 0, None, 1, True)


class QQP(ClassificationTask):
  """Quora Question Pair."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(QQP, self).__init__(config, "qqp", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    return self._load_glue(lines, split, 1 if split == "test" else 3,
                           2 if split == "test" else 4, 5, True)


class RTE(ClassificationTask):
  """Recognizing Textual Entailment."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(RTE, self).__init__(config, "rte", tokenizer,
                              ["entailment", "not_entailment"])

  def _create_examples(self, lines, split):
    return self._load_glue(lines, split, 1, 2, 3, True)


class QNLI(ClassificationTask):
  """Question NLI."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(QNLI, self).__init__(config, "qnli", tokenizer,
                               ["entailment", "not_entailment"])

  def _create_examples(self, lines, split):
    return self._load_glue(lines, split, 1, 2, 3, True)


class STS(RegressionTask):
  """Semantic Textual Similarity."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(STS, self).__init__(config, "sts", tokenizer, 0.0, 5.0)

  def _create_examples(self, lines, split):
    examples = []
    if split == "test":
      examples += self._load_glue(lines, split, -2, -1, None, True)
    else:
      examples += self._load_glue(lines, split, -3, -2, -1, True)
    if self.config.double_unordered and split == "train":
      examples += self._load_glue(
          lines, split, -3, -2, -1, True, len(examples), True)
    return examples
