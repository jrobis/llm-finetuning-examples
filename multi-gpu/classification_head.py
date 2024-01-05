import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import (AutoModelForCausalLM, AutoTokenizer, AdamW, default_data_collator,
                           get_linear_schedule_with_warmup, AutoModelForSequenceClassification)
import pandas as pd
import datasets
from datasets import Dataset
import accelerate

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from peft import get_peft_model, LoraConfig
from peft.utils.other import fsdp_auto_wrap_policy

from accelerate import Accelerator
from accelerate.logging import get_logger

import logging
import logging.handlers
from datetime import datetime
import numpy as np

import math
import time


## I HAD TO SET ... export NCCL_IGNORE_DISABLED_P2P=1 ... with a warning about NVLink
## export TOKENIZERS_PARALLELISM=false


def create_logger(cfg):
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

    return logger


def create_dataframe(cfg):
    df1 = pd.read_excel(f'{cfg.data.data_dir}/writing-eng-1.xlsx', header=1)
    df2 = pd.read_excel(f'{cfg.data.data_dir}/writing-eng-2.xlsx', header=0)
    df = pd.concat([df1, df2])
    df.reset_index(drop=True, inplace=True)
    df.columns = ['testId', 'prompt', 'response', 'comp', 'mech', 'expr', 'overall']
    df = df[df['response'].isna()==False]
    instruction = "Below is a question and answer pair from an English writing test. Determine on a scale from 0 (bad) to 13 (good) how well the answer uses english grammar."
    df['prompt'] = df.apply(lambda row: str(row['prompt']).strip(), axis=1)
    df['response'] = df.apply(lambda row: str(row['response']).strip(), axis=1)

    grouped = df.groupby('testId').agg({
        'prompt': list,
        'response': list,
        'overall': 'first'  # take the first overall score
    }).reset_index()

    # Convert prompt and response columns into the desired format
    grouped['prompt_response'] = grouped.apply(
        lambda row: [{'prompt': p, 'response': r} for p, r in zip(row['prompt'], row['response'])], axis=1
    )

    # Drop the now redundant columns
    grouped = grouped.drop(columns=['prompt', 'response'])

    first_str = """Generate a numerical score from 0-13 for the prompt and response below using the Rubric pasted below. The prompt and repsponse below are part of an English written test measuring the proficiency of the Candidate.
Rubric = [{Score: 1, Definition: "Candidate has no ability to write in the target language."},
{Score: 2, Definition: "Writing uses only isolated words. No knowledge of grammatical structures. Excessive spelling, punctuation, and/or vocabulary mistakes are present."},
{Score: 3, Definition: "Definition: Writing uses only isolated words or phrases. Grammar knowledge is very limited. Excessive spelling, punctuation, and/ or vocabulary mistakes are present."},
{Score: 4, Definition: "Writing uses simple sentences, words, and/ or phrases. Candidate displays very basic knowledge of grammar structures, but makes frequent mistakes."},
{Score: 5, Definition: "Definition: Writing uses simple language structures with no elaboration. Candidate displays some knowledge of grammar structures, but mistakes are present. Candidate is unable to effectively express opinions and/or explain procedures. Frequent spelling, punctuation, and/ or vocabulary mistakes are present."},
{Score: 6, Definition: "Definition: Writing uses basic structures to convey meaning, but no advanced or formal structures are used correctly. Candidate demonstrates a basic understanding of grammar structures, but many mistakes are present. Candidate is unable to effectively express opinions and/or explain procedures. Spelling, punctuation, and/ or vocabulary mistakes are present."},
{Score: 7, Definition: "Writing uses basic structures to convey meaning, but almost no advanced or formal structures are used correctly. Candidate demonstrates a basic understanding of grammar structures, but mistakes are present. Candidate might be unable to effectively express opinions and/ or explain procedures in a coherent manner. Spelling, punctuation, and vocabulary is good in areas of frequent usage, but mistakes are present in advanced areas.
"},
{Score: 8, Definition: "Writing uses basic structures to convey meaning, but few advanced or formal structures are used correctly. Candidate understands basic grammar structures, but mistakes are present in advanced areas. Candidate might have limited ability to express opinions and explain procedures in a coherent manner. Spelling, punctuation, and vocabulary is very good in areas of frequent usage, but mistakes are present in advanced areas that may confuse the reader"},
{Score: 9, Definition: "Definition: Writing uses basic and advanced structures to convey the meaning. Candidate understands basic and advanced grammar, but some mistakes are present. Candidate has basic ability to express opinions and explain procedures. Spelling, punctuation, and vocabulary is very good in areas of frequent usage, but mistakes are present in advanced areas that distract but do not confuse the reader."},
{Score: 10, Definition: "Writing structure is clear and concise, but lacks style and fluidity. Candidate understands basic and advanced grammar, but a few mistakes are present. Candidate is able to express opinions and explain procedures in an informal style. Spelling, punctuation, and/or vocabulary is very good in areas of frequent and infrequent usage, but mistakes are still present."},
{Score: 11, Definition: "Writing structure is clear and concise, but may lack style similar to that of a less-educated writer. Candidate use basic and advanced grammar correctly with
very minor errors. Candidate is able to express opinions and explain procedures, but may not use formal and informal styles effectively. Spelling, punctuation
and/or vocabulary mistakes are very few minor."},
{Score: 12, Definition: "Writing structure is equivalent to that of a well-educated writer. Candidate is able to express opinions and explain procedures in a way that demonstrates an
ability to write formal and informal styles. Grammar, spelling, punctuation, and/or vocabulary mistakes are very minor mistakes that a native speaker would
make."},
{Score: 13, Definition: "Writing structure is equivalent to that of a well-educated native writer. Complete range of linguistic nuance with no mistakes present."}
]

Prompts and Responses: """


    grouped['qa'] = grouped.apply(lambda row: first_str + str(row['prompt_response']), axis=1)
    return grouped


def create_accelerator(cfg):
    accelerator = Accelerator(log_with=cfg.tracking.report_to, project_dir=cfg.output_dir, mixed_precision='bf16', gradient_accumulation_steps=cfg.training.gradient_accumulation_steps)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return accelerator


def get_model_and_tokenizer(model_name):
    llama2 = AutoModelForSequenceClassification.from_pretrained(model_name,                                            
                                             torch_dtype=torch.bfloat16,
                                             num_labels=14, 
                                             trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          trust_remote_code=True, 
                                          use_fast=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    llama2.config.pad_token_id = llama2.config.eos_token_id

    return llama2, tokenizer
    # return None, tokenizer


def create_optimizer(model, cfg):
    return accelerate.utils.DummyOptim(model.parameters(), lr=cfg.training.learning_rate)


def create_dataloaders(df, tokenizer, cfg):
    if cfg.data.sample_size is not None:
        tmp_df = df.iloc[:cfg.data.sample_size]
    else:
        tmp_df = df
    qa = tmp_df['qa']
    scores = tmp_df['overall']

    # Convert text inputs to input tensors
    qa_tok = tokenizer(qa.tolist(), padding=True, truncation=True, return_tensors='pt', max_length=1540)
    input_ids = qa_tok['input_ids']
    attention_mask = qa_tok['attention_mask']

    # Convert scores to tensor
    # labels = torch.tensor(scores.values, dtype=torch.bfloat16)
    tmp_labels = torch.tensor(scores.values, dtype=torch.bfloat16)
    one_hot_list = list()

    def create_score_array(score):
        tmp_arr = np.zeros(14)
        tmp_arr[int(score)] = 1
        return tmp_arr
    
    for i in tmp_labels:
        score_arr = create_score_array(i)
        one_hot_list.append(score_arr)
    
    labels = torch.tensor(one_hot_list, dtype=torch.bfloat16)

    data_dict = {
        'input_ids': input_ids.tolist(),
        'attention_mask': attention_mask.tolist(),
        'label': labels.tolist()
    }

    dataset = Dataset.from_dict(data_dict)
    datasets = dataset.train_test_split(test_size=cfg.data.test_size)

    train_dataset = datasets["train"]
    test_dataset = datasets["test"]

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=default_data_collator,
        batch_size=cfg.training.train_batch_size,
        shuffle=True
    )

    eval_dataloader = DataLoader(
        test_dataset,
        collate_fn=default_data_collator,
        batch_size=cfg.training.eval_batch_size,
        shuffle=True
    )

    return train_dataloader, eval_dataloader


def get_size(model):
    print(f'{sum(p.numel() for p in model.parameters()):,}')


def main():

    # Create config and logger
    cfg = OmegaConf.load('conf/config.yaml')

    # Create accelerator
    accelerator = create_accelerator(cfg)
    accelerator.wait_for_everyone()
    
    # Create logger    
    logger = create_logger(cfg)
    logger.info('Config, Accelerator, Logger created.', main_process_only=True)

    # Import data
    df = create_dataframe(cfg)
    logger.info('Dataframe created.', main_process_only=True)
    logger.info(f'Length of Dataframe: {df.shape[0]:,}.')

    # Initialize model and tokenizer
    model, tokenizer = get_model_and_tokenizer(cfg.model.model_name)
    logger.info('Model and Tokenizer created.')

    # Transform data and create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(df, tokenizer, cfg)
    logger.info('Dataloaders created.')
    # for i, batch in enumerate(train_dataloader):
    #     # print(batch)
    #     logger.info(tokenizer.decode(batch['input_ids'][i]))
    #     if i > 3:
    #         break

    # return

    # Calculate total number of training steps
    total_num_steps = math.ceil(
            cfg.training.num_epochs * (len(train_dataloader) / cfg.training.gradient_accumulation_steps)
        )
    
    # Define loss function and optimizer
    optimizer = create_optimizer(model, cfg)
    lr_scheduler = accelerate.utils.DummyScheduler(optimizer, total_num_steps=total_num_steps, warmup_num_steps=cfg.training.lr_warmup_steps)
    logger.info('Loss func, Optimizer, LR Scheduler created.', main_process_only=True)


    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type = "SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
            ]
    )

    model = get_peft_model(model, peft_config)
    logger.info(model.print_trainable_parameters())

    (model, optimizer, train_dataloader, eval_dataloader, lr_scheduler) = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    logger.info('Accelerator prepared.', main_process_only=True)

    if accelerator.is_main_process:
        experiment_config = vars(cfg)
        accelerator.init_trackers(cfg.output_dir)
    
    progress_bar = tqdm(
        range(total_num_steps),
        disable=not accelerator.is_local_main_process,
    )
    
    # Set starting variables
    # epoch_durations = []
    eval_checkpoint_durations = []
    completed_steps = 0

     # Training loop
    for epoch in range(cfg.training.num_epochs):
        logger.info(f'Starting epoch: {epoch}')
        model.train()
        total_loss = 0
        train_losses = []
        eval_checkpoint_start_time = time.time()

        # Loop through training set
        for step, batch in enumerate(train_dataloader):
            if (step % cfg.training.gradient_accumulation_steps == 0):
                step_start_time = time.time()

            if step > cfg.training.max_train_steps_per_epoch:
                break

            # Forward pass
            with autocast(dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss              

            train_losses.append(
                accelerator.gather(loss.repeat(cfg.training.train_batch_size))
            )

            # We keep track of the loss at each step
            total_loss += loss.detach().float()
            loss = loss / cfg.training.gradient_accumulation_steps

            # Backward pass
            accelerator.backward(loss)

            # Complete step
            if (step % cfg.training.gradient_accumulation_steps == 0):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                train_losses_tensor = torch.cat(train_losses)
                train_loss = torch.mean(train_losses_tensor)
                num_gpus=8
                overall_batch_size = cfg.training.train_batch_size*cfg.training.gradient_accumulation_steps*num_gpus
                step_duration = time.time() - step_start_time
                records_per_second = overall_batch_size/step_duration
                accelerator.log({"train_loss": train_loss, "epoch": epoch, "step_duration": step_duration, "records_per_second": records_per_second}, step=completed_steps)
                logger.info(f'epoch: {epoch}; step {step}; train_loss: {train_loss}; step_duration {step_duration}; records_per_second {records_per_second}')
                
            # Run evaluation
            if step % cfg.training.eval_every == 0 and step > 0:
                model.eval()
                eval_losses = []

                model_preds = np.zeros(14)
                labels_arr = np.zeros(14)
            

                for _eval_step, eval_batch in enumerate(eval_dataloader):
                    if _eval_step > cfg.training.max_eval_steps:
                        break

                    for score in eval_batch['labels']:
                        position = np.argmax(np.array(score.cpu()))
                        labels_arr[int(position)] += 1
                    
                    with torch.no_grad():
                        with autocast(dtype=torch.bfloat16):
                            outputs = model(**eval_batch)
                            logits = outputs.logits

                            # Using argmax to get the most likely label index for each input in the batch
                            predicted_labels = torch.argmax(logits, dim=1).tolist()
                        
                            for score in predicted_labels:
                                model_preds[int(score)] += 1

                            # Calculate loss
                            loss = outputs.loss

                    eval_losses.append(
                        accelerator.gather(loss.repeat(cfg.training.eval_batch_size))
                    )
                
                losses = torch.cat(eval_losses)

                try:
                    eval_loss = torch.mean(losses)
                    eval_perplexity = math.exp(eval_loss)
                except OverflowError:
                    eval_perplexity = float("inf")

                def get_accuracy(true_labels, predictions):
                    if len(true_labels) != len(predictions):
                        raise ValueError("Both lists must have the same length")

                    correct = sum([1 for true, pred in zip(true_labels, predictions) if int(true) == int(pred)])
                    total = len(true_labels)
                    return correct / total

                # Example usage of get_accuracy:
                true_labels = torch.argmax(eval_batch['labels'], dim=1).tolist()
                accuracy = get_accuracy(true_labels, predicted_labels)
                # print(f"Accuracy: {acc * 100:.2f}%")

                eval_checkpoint_duration = time.time() - eval_checkpoint_start_time
                eval_checkpoint_durations.append(eval_checkpoint_duration)

                accelerator.log({"train_loss": train_loss, "epoch": epoch, 'eval_loss': eval_loss, 'eval_perplexity': eval_perplexity, 'accuracy': accuracy*100, 'eval_checkpoint_duration': eval_checkpoint_duration}, step=completed_steps)
                logger.info(
                    f"epoch {epoch}: step: {step} train_loss: {train_loss} eval_loss: {eval_loss} accuracy: {accuracy * 100:.2f} eval_perplexity: {eval_perplexity}  eval_checkpoint_duration: {eval_checkpoint_duration}"
                )
                logger.info(f"eval target scores {labels_arr}")
                logger.info(f"score distribution {model_preds}")

                eval_checkpoint_start_time = time.time()

                accelerator.wait_for_everyone()
                # # accelerator.save_state(cfg.output_dir)

                # unwrapped_model = accelerator.unwrap_model(model)
                # print('Unwrapped model')
                # print(f'Type of Unwrapped model: {type(unwrapped_model)}')
                # torch.save(unwrapped_model.state_dict(), cfg.output_dir+'/unwrap.pth')

                model.train()



if __name__ == "__main__":
    main()