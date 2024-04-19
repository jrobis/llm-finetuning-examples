import os
import math
import time

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

# import bittensor
import torch 
from torch.utils.data import DataLoader
import wandb
import datasets
from datasets import load_from_disk, concatenate_datasets
from accelerate import Accelerator
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer,
)

from accelerate.utils import set_seed

from torch import Tensor
import numpy as np

from tqdm.auto import tqdm
import logging
import accelerate
from accelerate.logging import get_logger

import json

import logging
import logging.handlers
from datetime import datetime


def check_cfg_and_load_defaults(cfg: DictConfig) -> DictConfig:

    subtensor = bittensor.subtensor(network=cfg.bittensor.network)
    if cfg.dataset.block_size is None:
        cfg.dataset.block_size = subtensor.validator_sequence_length
    if cfg.training.train_batch_size is None:
        cfg.training.train_batch_size = subtensor.validator_batch_size
    if cfg.training.eval_batch_size is None:
        cfg.training.eval_batch_size = subtensor.validator_batch_size

    return cfg

def load_dataset(cfg: DictConfig, logger):
    
    if cfg.model.name=='decapoda-research/llama-7b-hf':
        tokenized_dataset = load_from_disk("/mnt/share/the_pile/tokenized")
    
    else:
        status = True
        i = 0
        while status:
            file_name = cfg.dataset.file_name + "_" + str(i)
            data_file = os.path.join(cfg.dataset.data_dir, file_name)

            try:
                tokenized_dataset_batch = load_from_disk(data_file)
                tokenized_dataset_batch.set_format(type='pt')
                logger.info(f"loaded data from {data_file}.")
            except:
                status = False
                logger.info(f"{data_file} doesn't exist.")

            if i==0:
                tokenized_dataset = tokenized_dataset_batch
            else:
                tokenized_dataset = concatenate_datasets([tokenized_dataset, tokenized_dataset_batch])
            
            i += 1

    return tokenized_dataset

def load_tokenizer(cfg: DictConfig):

    if cfg.model.name=='decapoda-research/llama-7b-hf':
        tokenizer = LlamaTokenizer.from_pretrained(cfg.model.name, config="/mnt/share/llama-7b-hf/tokenizer_config.json")
        tokenizer.pad_token = cfg.tokenizer.pad_token
        # tokenizer.eos_token = cfg.tokenizer.eos_token
    else:

        if cfg.tokenizer.name is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.tokenizer.name, use_fast=cfg.tokenizer.use_fast
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                # cfg.model.name, use_fast=cfg.tokenizer.use_fast
                cfg.model.name
            )
        
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def create_accelerator(cfg: DictConfig) -> Accelerator:
    accelerator = (
        Accelerator(log_with=cfg.tracking.report_to, project_dir=cfg.output_dir)
        if cfg.tracking.enabled
        else Accelerator()
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return accelerator

def load_model(cfg: DictConfig, tokenizer):
    if cfg.model.name=='decapoda-research/llama-7b-hf':
        model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
        # , cache_dir='/mnt/share/.cache/huggingface/hub')
        model.resize_token_embeddings(len(tokenizer))
    else:
        if cfg.model.config_name is not None:
            config = AutoConfig.from_pretrained(cfg.model.config_name)
        else:
            config = AutoConfig.from_pretrained(cfg.model.name)

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            from_tf=bool(".ckpt" in cfg.model.name),
            config=config,
        )
        model.resize_token_embeddings(len(tokenizer))

    return model

def create_optimizer(cfg, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # LLaMA was throwing an error because second dict params was empty
    for i, _dict in enumerate(optimizer_grouped_parameters):
        if len(_dict['params'])< 1:
            optimizer_grouped_parameters.pop(i)
    
    return accelerate.utils.DummyOptim(optimizer_grouped_parameters, lr=cfg.training.learning_rate)
 
def attention_zeros(attn) -> Tensor:
    attn_zeros = attn
    idxs = np.random.randint(0, len(attn[0]-1), len(attn))
    
    for i, k in enumerate(zip(idxs, attn)):
        attn_zeros[i][k[1]][k[0]] = 0

    return attn_zeros

# def set_seeds(seed=17):
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)


def main():

    # Create config
    cfg = OmegaConf.load('conf/config.yaml')
    # cfg = check_cfg_and_load_defaults(cfg)

    # Create logger
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    current_time = datetime.now().strftime("-%m-%d-%Y-%H:%M:%S")    
    handler = logging.handlers.RotatingFileHandler(
        cfg.logfile + current_time, maxBytes=(1048576*5), backupCount=7
    )
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.logger.addHandler(handler)

    # Create accelerator
    accelerator = create_accelerator(cfg)
    accelerator.wait_for_everyone()

    

    # Set random seed
    # if cfg.training.seed is not None:
    #     logger.info(f"Setting random seed to {cfg.training.seed}")
    #     set_seed(cfg.training.seed)


    # Set device
    # device = accelerator.device
    

    print('Before tokenized datasets')

    # Set up dataloaders
    tokenized_datasets = load_dataset(cfg, logger)

    print('After tokenized datasets')
    if cfg.model.name=='decapoda-research/llama-7b-hf':
        tokenized_datasets = tokenized_datasets['train'].train_test_split(
            test_size=cfg.training.val_split_percent / 100)
    else:
        if "train" not in tokenized_datasets.column_names:
            tokenized_datasets = tokenized_datasets['train'].train_test_split(
                test_size=cfg.training.val_split_percent / 100)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    logger.info(f"train dataset shape {train_dataset.shape}")
    logger.info(f"eval dataset shape {eval_dataset.shape}")
    # print(f"test dataset shape {test_dataset.shape}")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=cfg.training.train_batch_size,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=cfg.training.eval_batch_size,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = (
            cfg.training.num_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        cfg.training.max_train_steps = (
            cfg.training.num_epochs * num_update_steps_per_epoch
        )
        # Afterwards we recalculate our number of training epochs
    cfg.training.num_epochs = math.ceil(
        cfg.training.max_train_steps / num_update_steps_per_epoch
    )
    

    # Create model and training objects
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, tokenizer)
    optimizer = create_optimizer(cfg, model)
    lr_scheduler = accelerate.utils.DummyScheduler(optimizer, total_num_steps=cfg.training.max_train_steps, warmup_num_steps=cfg.training.lr_warmup_steps)

    (model, optimizer, train_dataloader, 
        eval_dataloader, lr_scheduler) = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)


    # Set starting variables
    epoch_durations = []
    train_checkpoint_durations = []
    completed_steps = 0
    starting_epoch = 0
    resume_step = 0

    
    # Load from checkpoint
    if cfg.training.checkpoint.resume_from_checkpoint:

        with open(os.path.join(cfg.output_dir, cfg.training.checkpoint.checkpoint_file), "r") as infile:
            ckpt_meta_dict = json.load(infile)

        starting_epoch = ckpt_meta_dict['epoch']
        resume_step = ckpt_meta_dict['step']

        logger.info(f"starting epoch: {starting_epoch}")
        logger.info(f"resume step: {resume_step}")
        
        accelerator.wait_for_everyone()
        accelerator.load_state(cfg.output_dir)
        accelerator.wait_for_everyone()

        logger.info('\n\n\Model params')
        for name, param in model.named_parameters():
            logger.info(f"name: {name} - param: {param}")
            break
        

    # Initializing W&B tracker 
    if cfg.tracking.enabled and accelerator.is_main_process:
        experiment_config = vars(cfg)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = cfg.training.lr_scheduler
        accelerator.init_trackers("finetune_using_clm", experiment_config)

    
    # Logging accelerator information
    logger.info(accelerator.state, main_process_only=False)
    logger.info(OmegaConf.to_yaml(cfg))
        
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.training.num_epochs}")
    logger.info(
        f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")


    # Create progress bar for only 1 GPU on each machine
    progress_bar = tqdm(
        range(cfg.training.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )


    # Start training
    for epoch in range(starting_epoch, cfg.training.num_epochs):

        # Skip epoch
        if (cfg.training.checkpoint.resume_from_checkpoint 
            and starting_epoch is not None 
            and resume_step is not None 
            and epoch < starting_epoch):
            logger.info(f"skiping epoch: {epoch}")
            continue

        # Set vars at start of each epoch
        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        train_losses = []
        train_checkpoint_start_time = time.time()
        
        # Loop through training set
        for step, batch in enumerate(train_dataloader):
            if step > cfg.training.max_train_steps:
                        break
            # Skip step
            if (cfg.training.checkpoint.resume_from_checkpoint 
                and starting_epoch is not None 
                and resume_step is not None 
                and epoch < starting_epoch 
                and step < resume_step):
                logger.info(f"skiping step: {step}")
                completed_steps += 1
                continue
            
            # Set random token to 0 attention
            tmp_attn = batch['attention_mask']
            batch['attention_mask'] = attention_zeros(tmp_attn)

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            train_losses.append(
                accelerator.gather(loss.repeat(cfg.training.train_batch_size))
            )

            # We keep track of the loss at each epoch
            if cfg.tracking.enabled is True:
                total_loss += loss.detach().float()
            loss = loss / cfg.training.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)

            # Complete step
            if (
                    step % cfg.training.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

            # Run evalution
            if step % cfg.training.eval_every == 0:
                train_losses_tensor = torch.cat(train_losses)
                train_loss = torch.mean(train_losses_tensor)
                model.eval()
                eval_losses = []
                
                for _eval_step, eval_batch in enumerate(eval_dataloader):
                    if _eval_step > cfg.training.max_eval_steps:
                        break
                    with torch.no_grad():
                        outputs = model(**eval_batch)

                    loss = outputs.loss
                    eval_losses.append(
                        accelerator.gather(loss.repeat(cfg.training.eval_batch_size))
                    )

                losses = torch.cat(eval_losses)
                losses = losses[: len(eval_dataset)]
                try:
                    eval_loss = torch.mean(losses)
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                train_checkpoint_duration = time.time() - train_checkpoint_start_time
                train_checkpoint_durations.append(train_checkpoint_duration)
                
                accelerator.log({"train_loss": train_loss, "epoch": epoch, 'eval_loss': eval_loss, 'eval_perplexity': perplexity, 'train_checkpoint_duration': train_checkpoint_duration}, step=completed_steps)
                logger.info(
                    f"epoch {epoch}: eval_perplexity: {perplexity} train_loss: {train_loss} eval_loss: {eval_loss} 'train_checkpoint_duration': {train_checkpoint_duration} step: {completed_steps}"
                )

                train_checkpoint_start_time = time.time()
                model.train()

               
                
                # Save accelerator state
                accelerator.wait_for_everyone()
                accelerator.save_state(cfg.output_dir)

                # # Save model checkpoint
                # accelerator.wait_for_everyone()
                # unwrapped_model = accelerator.unwrap_model(model)
                # unwrapped_model.save_pretrained(
                #     cfg.output_dir,
                #     is_main_process=accelerator.is_main_process,
                #     save_function=accelerator.save,
                # )

                # if accelerator.is_main_process:
                #     tokenizer.save_pretrained(cfg.output_dir)

                ckpt_dict = {"epoch": epoch, "step": step}
                with open(os.path.join(cfg.output_dir, cfg.training.checkpoint.checkpoint_file), "w") as outfile:
                    json.dump(ckpt_dict, outfile)
            

        # Calculate and log loss at end of epoch
        train_loss = total_loss.item() / len(train_dataloader)

        if cfg.tracking.enabled is True:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": train_loss,
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        # Calculate and log epoch
        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)

        accelerator.log({"train_loss": train_loss, "epoch": epoch, 'eval_loss': eval_loss, 'eval_perplexity': perplexity, 'epoch_duration': epoch_duration}, step=completed_steps)
        logger.info(f"done epoch {epoch}")

        # Save accelerator state
        accelerator.wait_for_everyone()
        accelerator.save_state(cfg.output_dir)

        ckpt_dict = {"epoch": epoch, "step": step}
        with open(os.path.join(cfg.output_dir, cfg.training.checkpoint.checkpoint_file), "w") as outfile:
            json.dump(ckpt_dict, outfile)
        
        logger.info('\n\n\Model params')
        for name, param in model.named_parameters():
            logger.info(f"name: {name} - param: {param}")
            break

    # Calculate and log training run
    avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
    avg_train_checkpoint_runtime = sum(train_checkpoint_durations) / len(train_checkpoint_durations)

    logger.info({"avg epoch runtime (seconds)": avg_epoch_runtime})
    logger.info({"avg train checkpoint runtime (seconds)": avg_train_checkpoint_runtime})

    # Save accelerator state
    accelerator.wait_for_everyone()
    accelerator.save_state(cfg.output_dir)

    ckpt_dict = {"epoch": epoch, "step": step}
    with open(os.path.join(cfg.output_dir, cfg.training.checkpoint.checkpoint_file), "w") as outfile:
        json.dump(ckpt_dict, outfile)

    logger.info('\n\n\Model params')
    for name, param in model.named_parameters():
        logger.info(f"name: {name} - param: {param}")
        break


if __name__ == "__main__":
    main()