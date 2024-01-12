# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pdb
import os
import time
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import json
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb

from src.options import Options
from src import data, beir_utils, slurm, dist_utils, utils, contriever, finetuning_data, inbatch

import train

# hs : add 240103
from src import peft_model
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def finetuning(opt, model, optimizer, scheduler, tokenizer, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    if hasattr(model, "module"):
        eval_model = model.module
    else:
        eval_model = model
    eval_model = eval_model.get_encoder()

    train_dataset = finetuning_data.Dataset(
        datapaths=opt.train_data,
        negative_ctxs=opt.negative_ctxs,
        negative_hard_ratio=opt.negative_hard_ratio,
        negative_hard_min_idx=opt.negative_hard_min_idx,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        training=True,
    )
    collator = finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
    evaluate(opt, eval_model, tokenizer, tb_logger, step)

    epoch = 1

    model.train()
    prev_ids, prev_mask = None, None
    while step < opt.total_steps:
        logger.info(f"Start epoch {epoch}, number of batches: {len(train_dataloader)}")
        for i, batch in enumerate(train_dataloader):
            batch = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            step += 1

            train_loss, iter_stats = model(**batch, stats_prefix="train")
            train_loss.backward()

            if opt.optim == "sam" or opt.optim == "asam":
                optimizer.first_step(zero_grad=True)

                sam_loss, _ = model(**batch, stats_prefix="train/sam_opt")
                sam_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                wandb_info={}
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3f}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                        
                        wandb_info[k.replace('/','_')]=v
                        
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                wandb_info['train_epoch']=epoch
                wandb_info['train_step']=step
                wandb_info['lr']=float(f"{scheduler.get_last_lr()[0]:0.3g}")
                wandb_info['memory']=float(f"{torch.cuda.max_memory_allocated()//1e9}")

                # 🐝 2️⃣ Log metrics from your script to W&B
                if opt.local_rank==0:
                    wandb.log(wandb_info)
                logger.info(log)
                
                run_stats.reset()

            if step % opt.eval_freq == 0:
                train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
                evaluate(opt, eval_model, tokenizer, tb_logger, step)

                if step % opt.save_freq == 0 and dist_utils.get_rank() == 0:
                    if opt.use_peft:
                        utils.peft_save(
                            eval_model,
                            optimizer,
                            scheduler,
                            step,
                            opt,
                            opt.output_dir,
                            f"step-{step}",
                        )
                        
                    else:
                        utils.save(
                            eval_model,
                            optimizer,
                            scheduler,
                            step,
                            opt,
                            opt.output_dir,
                            f"step-{step}",
                        )
                model.train()

            if step >= opt.total_steps:
                break

        epoch += 1

    # Mark the run as finished
    wandb.finish()


def evaluate(opt, model, tokenizer, tb_logger, step):
    dataset = finetuning_data.Dataset(
        datapaths=opt.eval_data,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        training=False,
    )
    collator = finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    correct_samples, total_samples, total_step = 0, 0, 0
    all_q, all_g, all_n = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

            all_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], dim=0)
            all_mask = torch.cat([batch["g_mask"], batch["n_mask"]], dim=0)

            q_emb = model(input_ids=batch["q_tokens"], attention_mask=batch["q_mask"], normalize=opt.norm_query)
            all_emb = model(input_ids=all_tokens, attention_mask=all_mask, normalize=opt.norm_doc)

            g_emb, n_emb = torch.split(all_emb, [len(batch["g_tokens"]), len(batch["n_tokens"])])

            all_q.append(q_emb)
            all_g.append(g_emb)
            all_n.append(n_emb)

        all_q = torch.cat(all_q, dim=0)
        all_g = torch.cat(all_g, dim=0)
        all_n = torch.cat(all_n, dim=0)

        labels = torch.arange(0, len(all_q), device=all_q.device, dtype=torch.long)

        all_sizes = dist_utils.get_varsize(all_g)
        all_g = dist_utils.varsize_gather_nograd(all_g)
        all_n = dist_utils.varsize_gather_nograd(all_n)
        labels = labels + sum(all_sizes[: dist_utils.get_rank()])

        scores_pos = torch.einsum("id, jd->ij", all_q, all_g)
        scores_neg = torch.einsum("id, jd->ij", all_q, all_n)
        scores = torch.cat([scores_pos, scores_neg], dim=-1)

        argmax_idx = torch.argmax(scores, dim=1)
        sorted_scores, indices = torch.sort(scores, descending=True)
        isrelevant = indices == labels[:, None]
        rs = [r.cpu().numpy().nonzero()[0] for r in isrelevant]
        mrr = np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])

        acc = (argmax_idx == labels).sum() / all_q.size(0)
        acc, total = dist_utils.weighted_average(acc, all_q.size(0))
        mrr, _ = dist_utils.weighted_average(mrr, all_q.size(0))
        acc = 100 * acc

        message = []
        wandb_log={}
        if dist_utils.is_main():
            message = [f"eval acc: {acc:.2f}%", f"eval mrr: {mrr:.3f}"]
            logger.info(" | ".join(message))
            if tb_logger is not None:
                tb_logger.add_scalar(f"eval_acc", acc, step)
                tb_logger.add_scalar(f"mrr", mrr, step)

                wandb_log['eval_acc']=acc
                wandb_log['mrr']=mrr
                wandb_log['eval_step']=step

                if opt.local_rank==0:
                    wandb.log(wandb_log)


def main(opt):
    logger.info("Start")
    # wandb.login()
    
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        slurm.init_distributed_mode(opt)
        slurm.init_signal_handler()

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    step = 0

    if not opt.use_peft:
        retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_path, opt.pooling, opt.random_init)
    else:
        retriever, tokenizer, retriever_model_id = peft_model.create_and_prepare_model(opt)

    opt.retriever_model_id = retriever_model_id
    model = inbatch.InBatch(opt, retriever, tokenizer)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer, scheduler = utils.set_optim(opt, model)
    logger.info(utils.get_parameters(model))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = opt.dropout

    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    finetuning(opt, model, optimizer, scheduler, tokenizer, step)


if __name__ == "__main__":
    options = Options()
    opt = options.parse()

    if opt.local_rank==0:
        wandb.init(
            # Set the project where this run will be logged
            project="SmartPatent", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{opt.wandb_run_name}", 
            # Track hyperparameters and run metadata
            config=opt)
    
    main(opt)
