"""
这个程序定义了一个名为TuringNLRv3Config的配置类，用于存储和管理TuringNLRv3Model模型的配置。这个配置类包含了模型的各种参数，如词汇表大小、隐藏层的
大小、Transformer编码器的层数、注意力头的数量、中间层的大小、激活函数类型、dropout概率、最大序列长度、类型词汇表大小、初始化范围和层归一化中的
epsilon值等。

主要功能如下：

定义和初始化配置参数：类中的参数用于控制TuringNLRv3Model的结构和行为，比如隐藏层大小、注意力头数量等。当创建TuringNLRv3Config对象时，这些参数可以
根据需要被传入或使用默认值。

支持从配置文件加载参数：如果vocab_size参数是字符串（表示一个文件路径），程序会读取该文件并解析其中的配置参数，以设置TuringNLRv3Config对象的属性。

管理模型配置：这个类继承自PretrainedConfig类，可以用于保存和加载预训练模型的配置。这种设计使得模型配置可以被保存到文件中，方便后续的使用和复现。
"""

# coding=utf-8
""" TuringNLRv3 model configuration """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import sys
from io import open

from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)

TuringNLRv3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
}


class TuringNLRv3Config(PretrainedConfig):
    r"""
        :class:`~transformers.TuringNLRv3Config` is the configuration class to store the configuration of a
        `TuringNLRv3Model`.
        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `TuringNLRv3Model`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `TuringNLRv3Model`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = TuringNLRv3_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size=28996,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=6,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 source_type_id=0, 
                 target_type_id=1,
                 **kwargs):
        super(TuringNLRv3Config, self).__init__(**kwargs)
        if isinstance(vocab_size, str) or (sys.version_info[0] == 2
                                           and isinstance(vocab_size, unicode)):
            with open(vocab_size, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size, int):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.source_type_id = source_type_id
            self.target_type_id = target_type_id
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             " or the path to a pretrained model config file (str)")
