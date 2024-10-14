import os

# 用于操作文件和文件路径，可以用于跨平台处理文件路径
from pathlib import Path

import torch.multiprocessing as mp

from src.parameters import parse_args
from src.run import train, test

# setuplogging 函数通常用于设置项目的日志记录配置。
from src.utils import setuplogging

if __name__ == "__main__":

    setuplogging()

    # 细节：
    # range(2) 生成一个范围对象，返回 0 和 1。
    # str(_ + 1) 将 0 和 1 加 1，得到 1 和 2，并将它们转换为字符串。
    # ','.join(...) 将生成的字符串 1 和 2 通过逗号 , 连接在一起，形成字符串 '1,2'。
    #  结果：gpus 的值将是 '1,2'，表示使用编号为 1 和 2 的 GPU 设备。
    
    gpus = ','.join([str(_ + 1) for _ in range(2)])
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # 表示分布式训练的主节点就在当前机器上进行，通常用于单机多卡的分布式训练模式。
    os.environ['MASTER_ADDR'] = 'localhost'

    # 设置主节点的通信端口为 12355，用于分布式训练进程之间的数据通信。
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()

    # 获取当前的工作目录（Current Working Directory, CWD）
    print(os.getcwd())

    # log_steps 表示每隔多少步记录一次日志。这里设置为 5，即每训练或计算 5 步时记录一次日志信息。
    args.log_steps = 5
    
    args.world_size = 2  # GPU number
    args.mode = 'train'
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    cont = False
    if args.mode == 'train':
        print('-----------train------------')
        if args.world_size > 1:
            mp.freeze_support()
            mgr = mp.Manager()
            end = mgr.Value('b', False)
            mp.spawn(train,
                     args=(args, end, cont),
                     nprocs=args.world_size,
                     join=True)
        else:
            end = None
            train(0, args, end, cont)

    if args.mode == 'test':
        args.load_ckpt_name = "/data/workspace/Share/junhan/TopoGram_ckpt/dblp/topogram-pretrain-finetune-dblp-best3.pt"
        print('-------------test--------------')
        test(args)
