"""
1. 初始化与设置 (setup 和 cleanup 函数)
setup(rank, args):

初始化分布式环境，通过dist.init_process_group指定使用nccl后端（适合GPU），rank表示当前进程在分布式训练中的编号，world_size是总进程数。
设置当前进程所使用的GPU设备。
设置随机种子，确保结果可重复。
cleanup():

销毁分布式进程组，在所有训练或测试完成后调用，释放资源。

2. 模型加载 (load_bert 函数)
load_bert(args):
从配置文件或预训练模型路径中加载模型配置。
根据args.model_type的不同，加载不同的模型架构。
GraphFormersForNeighborPredict: 图神经网络模型用于邻居预测。
GraphSageMaxForNeighborPredict: 另一种图神经网络模型。
返回加载好的模型对象。

3. 训练函数 (train 函数)
train(local_rank, args, end, load):
初始化日志和分布式设置。
加载模型，如果启用了fp16（半精度浮点数），会使用GradScaler进行梯度缩放以防止精度损失。
如果需要，从指定的检查点（checkpoint）中加载模型权重。
如果world_size > 1，使用DistributedDataParallel将模型并行化，否则使用单一进程模型。
使用Adam优化器来更新模型的参数。
准备数据加载器，依据是否为多进程设置选择MultiProcessDataLoader或SingleProcessDataLoader。
在每个epoch的训练过程中：
进行前向和反向传播。
记录损失和优化器的状态。
保存最佳模型并进行验证和测试。

4. 测试函数 (test_single_process 和 test 函数)
test_single_process(model, args, mode):

测试模型在验证集或测试集上的表现。
不计算梯度（@torch.no_grad()），防止在测试过程中修改模型参数。
使用单进程数据加载器加载数据集，遍历每个batch，计算模型输出的评估指标。
返回测试的主要指标结果。
test(args):

加载模型，并加载指定的检查点。
调用test_single_process函数对模型进行测试。

5. 其他功能
使用日志记录模块logging来跟踪模型的训练和测试过程，包括训练损失、验证精度和保存的检查点等信息。
"""
import logging
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data_handler import DatasetForMatching, DataCollatorForMatching, SingleProcessDataLoader, \
    MultiProcessDataLoader
from src.models.tnlrv3.configuration_tnlrv3 import TuringNLRv3Config


def setup(rank, args):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    # Explicitly setting seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)


def cleanup():
    dist.destroy_process_group()


def load_bert(args):
    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    if args.model_type == "GraphFormers":
        from src.models.modeling_graphformers import GraphFormersForNeighborPredict
        model = GraphFormersForNeighborPredict(config)
        model.load_state_dict(torch.load(args.model_name_or_path, map_location="cpu")['model_state_dict'], strict=False)
        # model = GraphFormersForNeighborPredict.from_pretrained(args.model_name_or_path, config=config)
    elif args.model_type == "GraphSageMax":
        from src.models.modeling_graphsage import GraphSageMaxForNeighborPredict
        model = GraphSageMaxForNeighborPredict.from_pretrained(args.model_name_or_path, config=config)
    return model


def train(local_rank, args, end, load):
    try:
        # 如果是主进程，则设置日志记录
        if local_rank == 0:
            from src.utils import setuplogging
            setuplogging()  # 设置日志记录

        os.environ["RANK"] = str(local_rank)  # 设置当前进程的 rank（进程编号）
        setup(local_rank, args)  # 初始化设置，可能包括分布式训练的配置

        # 如果使用 FP16 精度训练，则导入相关模块并初始化 GradScaler
        if args.fp16:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()  # 用于管理 FP16 训练的精度

        model = load_bert(args)  # 加载模型（可能是BERT）
        logging.info('loading model: {}'.format(args.model_type))  # 打印日志：加载模型类型
        model = model.cuda()  # 将模型移到 GPU 上

        # 如果加载模型的检查点，则恢复模型的状态
        if load:
            model.load_state_dict(torch.load(args.load_ckpt_name, map_location="cpu"))  # 加载模型的权重
            logging.info('load ckpt:{}'.format(args.load_ckpt_name))  # 打印日志：加载的检查点路径

        # 如果是分布式训练，使用 DDP（分布式数据并行）封装模型
        if args.world_size > 1:
            ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            ddp_model = model

        # 初始化优化器
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.lr}])

        # 数据处理器（用于处理数据集的批次和对齐）
        data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                                token_length=args.token_length, random_seed=args.random_seed)

        loss = 0.0  # 初始化损失
        global_step = 0  # 初始化训练步数
        best_acc, best_count = 0.0, 0  # 初始化最好的准确率和计数器

        # 训练循环
        for ep in range(args.epochs):
            start_time = time.time()  # 记录每个 epoch 的开始时间
            ddp_model.train()  # 将模型设置为训练模式
            dataset = DatasetForMatching(file_path=args.train_data_path)  # 加载训练数据集
            if args.world_size > 1:  # 如果是多进程分布式训练
                end.value = False
                dataloader = MultiProcessDataLoader(dataset,
                                                    batch_size=args.train_batch_size,
                                                    collate_fn=data_collator,
                                                    local_rank=local_rank,
                                                    world_size=args.world_size,
                                                    global_end=end)
            else:
                dataloader = SingleProcessDataLoader(dataset, batch_size=args.train_batch_size,
                                                     collate_fn=data_collator, blocking=True)
            # 遍历数据加载器中的每个批次进行训练
            for step, batch in enumerate(dataloader):
                # 将数据移到 GPU（如果启用了 GPU）
                if args.enable_gpu:
                    for k, v in batch.items():
                        if v is not None:
                            batch[k] = v.cuda(non_blocking=True)

                # 使用混合精度训练（FP16）时，进行自动类型转换
                if args.fp16:
                    with autocast():  # 开启自动混合精度训练
                        batch_loss = ddp_model(**batch)
                else:
                    batch_loss = ddp_model(**batch)
                
                # 累积损失并进行反向传播
                loss += batch_loss.item()
                optimizer.zero_grad()
                if args.fp16:
                    scaler.scale(batch_loss).backward()  # 使用缩放的反向传播
                    scaler.step(optimizer)  # 更新优化器
                    scaler.update()  # 更新 scaler
                else:
                    batch_loss.backward()  # 普通反向传播
                    optimizer.step()  # 更新优化器

                global_step += 1

                # 每隔一定步数打印训练信息
                if local_rank == 0 and global_step % args.log_steps == 0:
                    logging.info(
                        '[{}] cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(
                            local_rank, time.time() - start_time, global_step, optimizer.param_groups[0]['lr'],
                                        loss / args.log_steps))
                    loss = 0.0  # 重置损失

                dist.barrier()  # 确保所有进程同步

            logging.info("train time:{}".format(time.time() - start_time))  # 打印训练时间

            if local_rank == 0:  # 主进程进行保存和验证
                # 保存当前 epoch 的模型
                ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename, ep + 1))
                torch.save(model.state_dict(), ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

                logging.info("Star validation for epoch-{}".format(ep + 1))  # 开始验证
                acc = test_single_process(model, args, "valid")  # 进行验证
                logging.info("validation time:{}".format(time.time() - start_time))
                if acc > best_acc:  # 如果验证准确率更好，则保存最佳模型
                    ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
                    torch.save(model.state_dict(), ckpt_path)
                    logging.info(f"Model saved to {ckpt_path}")
                    best_acc = acc  # 更新最佳准确率
                    best_count = 0  # 重置计数器
                else:
                    best_count += 1
                    if best_count >= 2:  # 如果验证连续两次没有提升，则停止训练并进行测试
                        start_time = time.time()
                        ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
                        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                        logging.info("Star testing for best")
                        acc = test_single_process(model, args, "test")  # 在测试集上进行测试
                        logging.info("test time:{}".format(time.time() - start_time))
                        exit()  # 退出训练
            dist.barrier()  # 确保所有进程同步

        # 训练结束后，加载最好的模型并进行最终测试
        if local_rank == 0:
            start_time = time.time()
            ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
            model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            logging.info("Star testing for best")
            acc = test_single_process(model, args, "test")
            logging.info("test time:{}".format(time.time() - start_time))
        dist.barrier()  # 确保所有进程同步
        cleanup()  # 清理资源

    except:
        # 如果出现异常，打印错误信息
        import sys
        import traceback
        error_type, error_value, error_trace = sys.exc_info()
        traceback.print_tb(error_trace)
        logging.info(error_value)  # 记录错误信息


@torch.no_grad()
def test_single_process(model, args, mode):
    assert mode in {"valid", "test"}
    model.eval()

    data_collator = DataCollatorForMatching(mlm=args.mlm, neighbor_num=args.neighbor_num,
                                            token_length=args.token_length, random_seed=args.random_seed)
    if mode == "valid":
        dataset = DatasetForMatching(file_path=args.valid_data_path)
        dataloader = SingleProcessDataLoader(dataset, batch_size=args.valid_batch_size, collate_fn=data_collator)
    elif mode == "test":
        dataset = DatasetForMatching(file_path=args.test_data_path)
        dataloader = SingleProcessDataLoader(dataset, batch_size=args.test_batch_size, collate_fn=data_collator)

    count = 0
    metrics_total = defaultdict(float)
    for step, batch in enumerate(dataloader):
        if args.enable_gpu:
            for k, v in batch.items():
                if v is not None:
                    batch[k] = v.cuda(non_blocking=True)

        metrics = model.test(**batch)
        for k, v in metrics.items():
            metrics_total[k] += v
        count += 1
    for key in metrics_total:
        metrics_total[key] /= count
        logging.info("mode: {}, {}:{}".format(mode, key, metrics_total[key]))
    model.train()
    return metrics_total['main']


def test(args):
    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.cuda()

    checkpoint = torch.load(args.load_ckpt_name, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logging.info('load ckpt:{}'.format(args.load_ckpt_name))

    test_single_process(model, args, "test")
