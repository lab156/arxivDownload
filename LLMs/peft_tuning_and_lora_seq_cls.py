from peft import LoraConfig, TaskType
from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          Trainer,
                         )
from peft import (get_peft_model,
                  get_peft_config,
                  get_peft_model_state_dict,
                  set_peft_model_state_dict,
                  PeftType,
                  PromptEncoderConfig,
                 )
import os, sys
import torch
from datetime import datetime as dt
from datasets import load_dataset, Dataset, load_metric
import evaluate
import torch
import numpy as np
import toml
import glob
import json
from tqdm import tqdm


currentdir = os.path.abspath(os.path.curdir)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,parentdir+'/embed') 
from classifier_trainer.trainer import stream_arxiv_paragraphs
import parsing_xml as px
import peep_tar as peep
from extract import Definiendum

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gen_cfg(**kwargs):
    # GET the parse args default values
    config_path = kwargs.get('configpath', 'config.toml')
    cfg = toml.load(config_path)['peft-p-tuning-lora']
    cfg.update(kwargs)

    # This is permanent storage
    cfg['base_dir'] = os.environ.get('PERMSTORAGE', '/media/hd1') 
    # This is temporary fast storage
    cfg['local_dir'] = os.environ.get('TEMPFASTSTORAGE',
            '/tmp/rm_me_finetuning')  

    if os.path.isdir(cfg['local_dir']):
        logger.info(f"Temp fast storage set to: {cfg['local_dir']}")
    else:
        logger.info(f"""Created directory for temp-fast-storage, 
                set to: {cfg['local_dir']}""")
        os.mkdir(cfg['local_dir'])

    # CREATE LOG FILE AND OBJECT
    hoy = dt.now()
    timestamp = hoy.strftime("%b-%d_%H-%M")
    cfg['timestamp'] = timestamp
    path_str = 'trained_models/finetuning/HFTransformers_' + timestamp
    cfg['save_path_dir'] = os.path.join(cfg['base_dir'], path_str)
    
    # xml_lst is too long to go in the config
    xml_lst = glob.glob(
        os.path.join(cfg['base_dir'], cfg['glob_data_source']))

    # Number of workers for dataloader
    #cfg['n_workers'] = cfg['n_workers']
    
    FHandler = logging.FileHandler(cfg['local_dir']+"/training.log")
    logger.addHandler(FHandler)

    ## get environment variable
    cfg['hf_transformers_cache'] = os.environ.get('TRANSFORMERS_CACHE', None)
    
    return xml_lst, cfg

def parse_args():
    '''
    parse args should be run before gen_cfg
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='',
        help="""Path to save the finetuned model, dir name only.""")
    parser.add_argument('--configpath', type=str, default='',
        help="""Path to config.toml file.""")
    parser.add_argument('--load8bit', action='store_true', default=False,
        help="""Load the model with 8 bit precision""")
    args = parser.parse_args()

    # make sure --savepath exists
    if args.savedir != '':
        os.makedirs(args.savedir, exist_ok=True)

    return vars(args)

def get_dataset(xml_lst, cfg):
    stream = stream_arxiv_paragraphs(xml_lst,
             samples=cfg['data_stream_batch_size'])

    all_data = []
    all_labels = []
    all_texts = []
    for s in stream:
        try:
            #all_data += list(zip(s[0], s[1]))
            all_texts += s[0]
            all_labels += s[1]
        except IndexError:
            logger.warning('Index error in the data stream.')
    data_dict = {
        'text': all_texts,
        'label': all_labels
    }
    ds = Dataset.from_dict(data_dict)
    return ds

def compute_metrics(eval_pred): 
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1") 

    logits, labels = eval_pred
    #import pdb
    #pdb.set_trace()
    #predictions = np.argmax(logits[0], axis=-1) # used for mt0 models
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions,
           references=labels)["accuracy"]           
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}


def make_label_int(examples):
    # Convert the entries in the 'labels' to int
    # run with the .map() method
    outs = examples
    outs['new_label'] = [int(f) for f in examples['label']]
    return outs

def main():
    args = parse_args()
    xml_lst, cfg = gen_cfg(**args)
    assert len(xml_lst) > 0, 'Empty xml_lst'

    ds = get_dataset(xml_lst, cfg)
    ds = ds.train_test_split(test_size=0.1, shuffle=True)

    if any(k in cfg['checkpoint'] for k in ("gpt", "opt", "bloom")):
            padding_side = "left"
    else:
            padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(cfg['checkpoint'], 
                                        padding_side=padding_side,
                                     padding=True, 
                                     truncation=True, 
                                     max_length=cfg['max_length'],)

    if getattr(tokenizer, "pad_token_id") is None:
        print('Added pad_token_id attribute to tokenizer')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["text"], 
                           truncation=True,
                           max_length=cfg['max_length'])
        return outputs

    tokenized_datasets = ds.map(
        tokenize_function,
        batched=True,
        #remove_columns=["idx", "sentence1", "sentence2"],
        remove_columns=["text"],
    )

    # Convert labels from float to int
    tokenized_datasets = tokenized_datasets.map(
        make_label_int,
        batched=True,
        remove_columns=['label']
    )

    tokenized_datasets = tokenized_datasets.rename_column("new_label", "labels")
    

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, 
                                            padding="longest")
    
    if cfg['peft_conf'] == 'p-tuning':
    # Config for p-Tuning
        peft_config = PromptEncoderConfig(task_type="SEQ_CLS",
                                          num_virtual_tokens=20,
                                          encoder_hidden_size=128)
     
    else:
        peft_config = LoraConfig(task_type="SEQ_CLS", 
                                 inference_mode=False,
                                 r=16,
                                 lora_alpha=32,
                                 lora_dropout=0.1)

    model = AutoModelForSequenceClassification.from_pretrained(
            cfg['checkpoint'], return_dict=True)

    if 'gpt' in cfg['checkpoint']:
        model.config.pad_token_id = model.config.eos_token_id

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    output_dir = os.path.join(cfg['savedir'],
                        f"{cfg['checkpoint'].replace('/', '-')}-peft-lora"
                         )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        num_train_epochs=cfg['num_epochs'],
        weight_decay=cfg['weight_decay'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == '__main__':
    main()
