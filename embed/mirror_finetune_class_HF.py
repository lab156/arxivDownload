import tensorflow as tf
from datasets import load_dataset, Dataset, DatasetDict 
import pandas as pd
import sys
import os
import glob
import toml
import numpy as np
import logging
from datetime import datetime as dt
import time
#import pdb

from transformers import (AutoTokenizer,
                         TFAutoModelForSequenceClassification,
                         DataCollatorWithPadding,
                         TFPreTrainedModel,)

# keras for training
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
import sklearn.metrics as metrics

#currentdir = os.path.abspath(os.path.curdir)
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0,parentdir) 
#sys.path.insert(0,parentdir+'/embed') 
import sys,inspect
currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from classifier_trainer.trainer import stream_arxiv_paragraphs

#from train_lstm import find_best_cutoff

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gen_cfg(**kwargs):
    # GET the parse args default values
    config_path = kwargs.get('configpath', 'config.toml')
    cfg = toml.load(config_path)['finetuning']
    cfg.update(kwargs)

    # This is permanent storage
    cfg['base_dir'] = os.environ.get('PERMSTORAGE', '/media/hd1') 
    # This is temporary fast storage
    cfg['local_dir'] = os.environ.get('TEMPFASTSTORAGE',
            '/tmp/rm_me_finetuning')  

    # CREATE LOG FILE AND OBJECT
    hoy = dt.now()
    timestamp = hoy.strftime("%b-%d_%H-%M")
    cfg['timestamp'] = timestamp
    path_str = 'trained_models/finetuning/HFTransformers_' + timestamp
    cfg['save_path_dir'] = os.path.join(cfg['base_dir'], path_str)
    
    # xml_lst is too long to go in the config
    xml_lst = glob.glob(
        os.path.join(cfg['base_dir'], cfg['glob_data_source']))
    
    FHandler = logging.FileHandler(cfg['local_dir']+"/training.log")
    logger.addHandler(FHandler)
    
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
    
def prepare_data(ds, cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg['checkpoint'])

    def tok_function(example):
    # This function can be used with the Dataset.map() method
        return tokenizer(example['text'], truncation=True)

    tkn_data = ds.map(tok_function, batched=True)
    
    tkn_data = tkn_data.select(range(int(cfg['shrink_data_factor']*len(tkn_data))))
    temp1_dd = tkn_data.train_test_split(test_size=0.1, shuffle=True)
    temp2_dd = temp1_dd['train'].train_test_split(test_size=0.1, shuffle=True)

    tkn_data = DatasetDict({
        'train': temp2_dd['train'],
        'test': temp1_dd['test'],
        'valid': temp2_dd['test'],
    })

    def add_missing_token_type(ex):
        # return a zero array of the same length as attention_mask
        ex['token_type_ids'] = [len(x)*[0] for x in ex['attention_mask']]
        return ex

    column_lst = ['attention_mask', 'input_ids', 'token_type_ids']
    if cfg['checkpoint'].startswith('roberta'):
        logger.info('fixing missing token_type_ids for roberta')
        #tkn_data = tkn_data.map(add_missing_token_type, batched=True)
        column_lst = ['attention_mask', 'input_ids'] 
    if cfg['checkpoint'] == 'gpt2':
        column_lst = ['attention_mask', 'input_ids'] 
        assert tokenizer.pad_token is None, "tokenizer.pad_token is not None"
        tokenizer.pad_token = tokenizer.eos_token

    
    # This function does no accept the return_tensors argument.
    try:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                return_tensors='tf')
    except TypeError:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Take care of everyting using `to_tf_dataset()`
    tf_train_data = tkn_data['train'].to_tf_dataset(
           columns=column_lst,
           label_cols=['label'],
           shuffle=True,
           collate_fn=data_collator,
           batch_size=cfg['batch_size'] )

    tf_valid_data = tkn_data['valid'].to_tf_dataset(
           columns=column_lst,
           label_cols=['label'],
           shuffle=True,
           collate_fn=data_collator,
           batch_size=cfg['batch_size'])

    tf_test_data = tkn_data['test'].to_tf_dataset(
           columns=column_lst,
           label_cols=['label'],
           shuffle=False,
           collate_fn=data_collator,
           batch_size=cfg['batch_size'])
    
    return (tf_train_data, 
            tf_valid_data,
            tf_test_data,
            tokenizer,
           )

def make_HF_model(cfg, tokenizer=None):
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=cfg['initial_lr'], 
        end_learning_rate=cfg['end_lr'], 
        decay_steps=cfg['num_train_steps']
    )

    opt = Adam(learning_rate=lr_scheduler)

    #reload the model to change the optimizer
    model = TFAutoModelForSequenceClassification.from_pretrained(
                cfg['checkpoint'], num_labels=2)

    if cfg['checkpoint'] == 'gpt2':
        assert tokenizer is not None, "tokenizer should be set as argument not None"
        logger.info(
        f"""model.config.pad_token_id is {model.config.pad_token_id},
          changing to tokenizer.eos_token""")
        model.config.pad_token_id = tokenizer.eos_token_id

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #loss = tf.keras.losses.BinaryCrossentropy()

    model.compile(optimizer=opt,
                 loss=loss,
                 metrics=['accuracy'])
    return model

def find_best_cutoff(model, preds, test):
    '''
    model: is an instance with predict attribute
    test_seq: has same shape and format as the training sequence
    '''
    f1_max = 0.0; opt_prob = None
    #pred_data = model.predict(test_seq)
    pred_data = preds

    for thresh in np.arange(0.1, 0.901, 0.1):
        thresh = np.round(thresh, 2)
        f1 = metrics.f1_score(test, (pred_data > thresh).astype(int))
        #print('F1 score at threshold {} is {}'.format(thresh, f1))
        
        if f1 > f1_max:
            f1_max = f1
            opt_prob = thresh
    return (opt_prob, f1_max)

def main():
    # Training
    args = parse_args()
    xml_lst, cfg = gen_cfg(**args)
    assert len(xml_lst) > 0, 'Empty xml_lst'
    checkpoint = cfg['checkpoint']

    xla_gpu_lst = tf.config.list_physical_devices("XLA_GPU")
    logger.info(f'List of XLA GPUs: {xla_gpu_lst}')
    
    ds = get_dataset(xml_lst, cfg)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    cfg['batch_size'] = mirrored_strategy.num_replicas_in_sync * cfg['batch_size']

    print('num_replicas_in_sync=', mirrored_strategy.num_replicas_in_sync)
    logger.info(f'num_replicas_in_sync= {mirrored_strategy.num_replicas_in_sync}')

    
    (tf_train_data, 
    tf_valid_data,
    tf_test_data,
    tokenizer,
    ) = prepare_data(ds, cfg)
    
    cfg['num_train_steps'] = len(tf_train_data)*cfg['num_epochs']

    
    with mirrored_strategy.scope():
        # define the model and compile
        model = make_HF_model(cfg, tokenizer=tokenizer)
    # fit should be out of the the with scope
    model.fit( tf_train_data, validation_data=tf_valid_data, epochs=cfg['num_epochs'])
    
    #import pdb
    #pdb.set_trace()
    now = dt.now()
    try:
        #assert False
        preds = model.predict(tf_test_data)#['logits']
        class_preds = np.argmax(preds[0], axis=1)
        now = (dt.now() - now)
        logger.info(f"It took {now} secs to get out of predict.")
        
        targets = []
        for b in tf_test_data.as_numpy_iterator():
            targets.extend(list(b[1])) 

        opt_prob, f1_max = find_best_cutoff(model, class_preds, targets)

        logger.info(f"{opt_prob=} and {f1_max=}")
        print(f"{opt_prob=} and {f1_max=}")
        cfg['opt_thresh'] = opt_prob
        cfg['f1_max'] = f1_max
        metric_str = metrics.classification_report(
                (class_preds > opt_prob).astype(int), targets)
        print(metric_str)
        logger.info(metric_str)
    except tf.errors.ResourceExhaustedError as e:
        logger.info(f"Caught OOM error {e} \n proceed to save the model")
    except tf.errors.InvalidArgumentError as e:
        logger.info(f"Caught Invalid argument error {e} \n proceed to save the model")
        it = iter(tf_test_data)
        example = next(it)
        print(example)
        example = next(it)
        print(example)
        test_cnt = 0
        for t in tf_test_data:
            test_cnt += 1
            print(t[0]['input_ids'].shape)
        print(f'{test_cnt=}')
        

    #Save the model
    if cfg['savedir'] != '':
        print(f"Saving to {cfg['savedir']}")
        model.save_pretrained(save_directory=cfg['savedir'])
    else:
        logger.warning(
        "cfg['savedir'] is empty string, not saving model.")
    logger.info(cfg)


if __name__ == "__main__":
    main()
