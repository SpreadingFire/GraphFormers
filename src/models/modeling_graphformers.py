"""
这个程序定义了一些基于深度学习的类和方法，用于实现图神经网络的预训练模型和邻居预测任务。以下是对这个程序的中文解释。

1. 定义GraphTuringNLRPreTrainedModel类
GraphTuringNLRPreTrainedModel类继承自TuringNLRv3PreTrainedModel，用于加载预训练的图神经网络模型。类中的from_pretrained方法主要用于从缓存或者本地文件中加载预训练模型的权重。

方法解析：
from_pretrained方法检查指定的路径是否存在预训练模型的权重文件，如果存在则加载这些权重并进行转换。
如果模型的位置信息嵌入需要重新初始化，则会根据配置文件中指定的最大位置嵌入数调整嵌入矩阵的大小。
如果权重文件中的相对位置偏置权重尺寸不符合要求，则会进行调整。

2. 定义GraphAggregation类
GraphAggregation类继承自BertSelfAttention，实现了图注意力机制中的节点聚合操作。

方法解析：
forward方法接受输入的隐藏状态、注意力掩码以及相对位置偏置信息，计算新的节点表示（station_embed）。

3. 定义GraphBertEncoder类
GraphBertEncoder类继承自nn.Module，用于实现图卷积编码器，它通过多个BertLayer层进行堆叠来编码输入的图结构数据。

方法解析：
forward方法接受输入的隐藏状态和注意力掩码，逐层更新隐藏状态，并可以选择输出所有隐藏状态和注意力权重。

4. 定义GraphFormers类
GraphFormers类继承自TuringNLRv3PreTrainedModel，用于图神经网络的编码和位置偏置计算。

方法解析：
forward方法首先对输入进行嵌入，然后构建图节点之间的相对位置偏置和掩码信息，最后通过GraphBertEncoder类来获取最终编码结果。

5. 定义GraphFormersForNeighborPredict类
GraphFormersForNeighborPredict类继承自GraphTuringNLRPreTrainedModel，主要用于图节点的邻居预测任务。

方法解析：
infer方法：从输入数据中推理得到节点的嵌入表示。
test方法：在测试集上计算预测精度、AUC、MRR、NDCG等指标。
forward方法：执行前向传播，并计算交叉熵损失，用于模型的训练。
总结
这个程序实现了一个图神经网络的框架，包含从预训练模型中加载权重、对输入图进行编码和聚合操作、以及在图邻居预测任务上的训练和评估过程。该框架使用了深度学习和注意力机制，适用于大规模图数据的表示学习和下游任务的预测。

——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
这个程序将Transformer模型和图神经网络（GNN）结合在一起，主要体现在以下几个方面：

1. 基于Transformer的图编码器
程序中的GraphBertEncoder类和GraphFormers类将Transformer的注意力机制与图结构数据结合起来，对图数据进行编码。

GraphBertEncoder类
GraphBertEncoder类继承自nn.Module，使用了多个BertLayer层来编码图数据。每个BertLayer层实际上是一个标准的Transformer层（包含自注意力机制和前馈神经网络），用于处理输入图节点及其邻居节点之间的关系。
在forward方法中，编码器会根据图的结构更新节点的表示，同时通过GraphAggregation类来聚合节点的特征。这种设计将Transformer的多头自注意力机制用于图节点之间的信息传播和特征聚合。
GraphAggregation类
GraphAggregation类继承自BertSelfAttention，在图数据上实现了多头注意力机制。forward方法中的multi_head_attention函数通过注意力机制计算图节点之间的相似性，并基于此更新节点的表示。
这种方式相当于将图结构的拓扑信息嵌入到Transformer的注意力计算中，从而增强了图节点间的关系建模能力。

2. 相对位置偏置（Relative Position Bias）的使用
在GraphFormers类的forward方法中，引入了相对位置偏置（relative position bias），用来表示图节点之间的相对距离和位置关系。
relative_position_bucket函数用于计算相对位置桶（relative position bucket），然后通过self.rel_pos_bias线性层映射到注意力权重，这种方式在Transformer的自注意力机制中加入了图的位置信息。
这种相对位置偏置在图神经网络中尤为重要，因为图的节点之间存在复杂的相对位置和连接关系，通过这种方式可以更好地捕捉图的结构信息。

3. 图聚合操作与Transformer层的结合
在GraphBertEncoder的forward方法中，结合了图结构的自注意力聚合和Transformer编码层。在每层BertLayer之后，会通过GraphAggregation类中的multi_head_attention来计算每个节点的嵌入更新，考虑节点间的关系。
通过这种设计，每一层都会将图的结构信息与Transformer的表示学习能力结合起来，逐层提升节点的特征表示。

4. 基于节点邻居预测的任务实现
GraphFormersForNeighborPredict类中，infer和test方法通过图Transformer模型来推理节点的嵌入表示和预测邻居关系。模型通过输入查询节点和其邻居节点的序列来进行计算，将Transformer的能力用于图数据上的下游任务。
这种任务使用了Transformer模型来进行图上的节点分类或者关系预测任务，这本身就是一种将Transformer应用于图数据的场景。
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tnlrv3.convert_state_dict import get_checkpoint_from_transformer_cache, state_dict_convert
from src.models.tnlrv3.modeling import TuringNLRv3PreTrainedModel, logger, BertSelfAttention, BertLayer, WEIGHTS_NAME, \
    BertEmbeddings, relative_position_bucket
from src.utils import roc_auc_score, mrr_score, ndcg_score


class GraphTuringNLRPreTrainedModel(TuringNLRv3PreTrainedModel):
    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path, reuse_position_embedding=None,
            replace_prefix=None, *model_args, **kwargs,
    ):
        model_type = kwargs.pop('model_type', 'tnlrv3')
        if model_type is not None and "state_dict" not in kwargs:
            if model_type in cls.supported_convert_pretrained_model_archive_map:
                pretrained_model_archive_map = cls.supported_convert_pretrained_model_archive_map[model_type]
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    state_dict = get_checkpoint_from_transformer_cache(
                        archive_file=pretrained_model_archive_map[pretrained_model_name_or_path],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        pretrained_model_archive_map=pretrained_model_archive_map,
                        cache_dir=kwargs.get("cache_dir", None), force_download=kwargs.get("force_download", None),
                        proxies=kwargs.get("proxies", None), resume_download=kwargs.get("resume_download", None),
                    )
                    state_dict = state_dict_convert[model_type](state_dict)
                    kwargs["state_dict"] = state_dict
                    logger.info("Load HF ckpts")
                elif os.path.isfile(pretrained_model_name_or_path):
                    state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                elif os.path.isdir(pretrained_model_name_or_path):
                    state_dict = torch.load(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME),
                                            map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                else:
                    raise RuntimeError("Not fined the pre-trained checkpoint !")

        if kwargs["state_dict"] is None:
            logger.info("TNLRv3 does't support the model !")
            raise NotImplementedError()

        config = kwargs["config"]
        state_dict = kwargs["state_dict"]
        # initialize new position embeddings (From Microsoft/UniLM)
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict:
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                max_range = config.max_position_embeddings if reuse_position_embedding else old_vocab_size
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[shift: shift + delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " % (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding

        # initialize new rel_pos weight
        _k = 'bert.rel_pos_bias.weight'
        if _k in state_dict and state_dict[_k].shape[1] != (config.rel_pos_bins + 2):
            logger.info(
                f"rel_pos_bias.weight.shape[1]:{state_dict[_k].shape[1]} != config.bus_num+config.rel_pos_bins:{config.rel_pos_bins + 2}")
            old_rel_pos_bias = state_dict[_k]
            new_rel_pos_bias = torch.cat(
                [old_rel_pos_bias, old_rel_pos_bias[:, -1:].expand(old_rel_pos_bias.size(0), 2)], -1)
            new_rel_pos_bias = nn.Parameter(data=new_rel_pos_bias, requires_grad=True)
            state_dict[_k] = new_rel_pos_bias.data
            del new_rel_pos_bias

        if replace_prefix is not None:
            new_state_dict = {}
            for key in state_dict:
                if key.startswith(replace_prefix):
                    new_state_dict[key[len(replace_prefix):]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]
            kwargs["state_dict"] = new_state_dict
            del state_dict

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class GraphAggregation(BertSelfAttention):
    def __init__(self, config):
        super(GraphAggregation, self).__init__(config)
        self.output_attentions = False

    def forward(self, hidden_states, attention_mask=None, rel_pos=None):
        query = self.query(hidden_states[:, :1])  # B 1 D
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        station_embed = self.multi_head_attention(query=query,
                                                  key=key,
                                                  value=value,
                                                  attention_mask=attention_mask,
                                                  rel_pos=rel_pos)[0]  # B 1 D
        station_embed = station_embed.squeeze(1)

        return station_embed


class GraphBertEncoder(nn.Module):
    def __init__(self, config):
        super(GraphBertEncoder, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.graph_attention = GraphAggregation(config=config)

    def forward(self,
                hidden_states,
                attention_mask,
                node_mask=None,
                node_rel_pos=None,
                rel_pos=None):

        all_hidden_states = ()
        all_attentions = ()

        all_nodes_num, seq_length, emb_dim = hidden_states.shape
        batch_size, _, _, subgraph_node_num = node_mask.shape

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > 0:

                hidden_states = hidden_states.view(batch_size, subgraph_node_num, seq_length, emb_dim)  # B SN L D
                cls_emb = hidden_states[:, :, 1].clone()  # B SN D
                station_emb = self.graph_attention(hidden_states=cls_emb, attention_mask=node_mask,
                                                   rel_pos=node_rel_pos)  # B D

                # update the station in the query/key
                hidden_states[:, 0, 0] = station_emb
                hidden_states = hidden_states.view(all_nodes_num, seq_length, emb_dim)

                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, rel_pos=rel_pos)

            else:
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[::subgraph_node_num, :, :, 0] = -10000.0
                layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask, rel_pos=rel_pos)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class GraphFormers(TuringNLRv3PreTrainedModel):
    def __init__(self, config):
        super(GraphFormers, self).__init__(config=config)
        self.config = config
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = GraphBertEncoder(config=config)

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins + 2,
                                          config.num_attention_heads,
                                          bias=False)
        else:
            self.rel_pos_bias = None

    def forward(self,
                input_ids,
                attention_mask,
                neighbor_mask=None):
        all_nodes_num, seq_length = input_ids.shape
        batch_size, subgraph_node_num = neighbor_mask.shape

        embedding_output, position_ids = self.embeddings(input_ids=input_ids)

        # Add station attention mask
        station_mask = torch.zeros((all_nodes_num, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 1+L
        attention_mask[::(subgraph_node_num), 0] = 1.0  # only use the station for main nodes

        node_mask = (1.0 - neighbor_mask[:, None, None, :]) * -10000.0
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        if self.config.rel_pos_bins > 0:
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.config.rel_pos_bins,
                                               max_distance=self.config.max_rel_pos)

            # rel_pos: (N,L,L) -> (N,1+L,L)
            temp_pos = torch.zeros(all_nodes_num, 1, seq_length, dtype=rel_pos.dtype, device=rel_pos.device)
            rel_pos = torch.cat([temp_pos, rel_pos], dim=1)
            # rel_pos: (N,1+L,L) -> (N,1+L,1+L)
            station_relpos = torch.full((all_nodes_num, seq_length + 1, 1), self.config.rel_pos_bins,
                                        dtype=rel_pos.dtype, device=rel_pos.device)
            rel_pos = torch.cat([station_relpos, rel_pos], dim=-1)

            # node_rel_pos:(B:batch_size, Head_num, neighbor_num+1)
            node_pos = self.config.rel_pos_bins + 1
            node_rel_pos = torch.full((batch_size, subgraph_node_num), node_pos, dtype=rel_pos.dtype,
                                      device=rel_pos.device)
            node_rel_pos[:, 0] = 0
            node_rel_pos = F.one_hot(node_rel_pos,
                                     num_classes=self.config.rel_pos_bins + 2).type_as(
                embedding_output)
            node_rel_pos = self.rel_pos_bias(node_rel_pos).permute(0, 2, 1)  # B head_num, neighbor_num
            node_rel_pos = node_rel_pos.unsqueeze(2)  # B head_num 1 neighbor_num

            # rel_pos: (N,Head_num,1+L,1+L)
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins + 2).type_as(
                embedding_output)
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        else:
            node_rel_pos = None
            rel_pos = None

        # Add station_placeholder
        station_placeholder = torch.zeros(all_nodes_num, 1, embedding_output.size(-1)).type(
            embedding_output.dtype).to(embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 1+L D

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            node_mask=node_mask,
            node_rel_pos=node_rel_pos,
            rel_pos=rel_pos)

        return encoder_outputs


class GraphFormersForNeighborPredict(GraphTuringNLRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = GraphFormers(config)
        self.init_weights()

    def infer(self, input_ids_node_and_neighbors_batch, attention_mask_node_and_neighbors_batch,
              mask_node_and_neighbors_batch):
        B, N, L = input_ids_node_and_neighbors_batch.shape
        D = self.config.hidden_size
        input_ids = input_ids_node_and_neighbors_batch.view(B * N, L)
        attention_mask = attention_mask_node_and_neighbors_batch.view(B * N, L)
        hidden_states = self.bert(input_ids, attention_mask, mask_node_and_neighbors_batch)
        last_hidden_states = hidden_states[0]
        cls_embeddings = last_hidden_states[:, 1].view(B, N, D)  # [B,N,D]
        node_embeddings = cls_embeddings[:, 0, :]  # [B,D]
        return node_embeddings

    def test(self, input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
             mask_query_and_neighbors_batch, \
             input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch, mask_key_and_neighbors_batch,
             **kwargs):
        query_embeddings = self.infer(input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                                      mask_query_and_neighbors_batch)
        key_embeddings = self.infer(input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch,
                                    mask_key_and_neighbors_batch)
        scores = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)

        predictions = torch.argmax(scores, dim=-1)
        acc = (torch.sum((predictions == labels)) / labels.shape[0]).item()

        scores = scores.cpu().numpy()
        labels = F.one_hot(labels).cpu().numpy()
        auc_all = [roc_auc_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        auc = np.mean(auc_all)
        mrr_all = [mrr_score(labels[i], scores[i]) for i in range(labels.shape[0])]
        mrr = np.mean(mrr_all)
        ndcg_all = [ndcg_score(labels[i], scores[i], labels.shape[1]) for i in range(labels.shape[0])]
        ndcg = np.mean(ndcg_all)

        return {
            "main": acc,
            "acc": acc,
            "auc": auc,
            "mrr": mrr,
            "ndcg": ndcg
        }

    def forward(self, input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                mask_query_and_neighbors_batch, \
                input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch, mask_key_and_neighbors_batch,
                **kwargs):
        query_embeddings = self.infer(input_ids_query_and_neighbors_batch, attention_mask_query_and_neighbors_batch,
                                      mask_query_and_neighbors_batch)
        key_embeddings = self.infer(input_ids_key_and_neighbors_batch, attention_mask_key_and_neighbors_batch,
                                    mask_key_and_neighbors_batch)
        score = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1))
        labels = torch.arange(start=0, end=score.shape[0], dtype=torch.long, device=score.device)
        loss = F.cross_entropy(score, labels)
        return loss
