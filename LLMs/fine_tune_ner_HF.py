import sys
sys.path = [p for p in sys.path if 'luis' not in p]
import numpy as np
import tensorflow as tf
print('Using TF version: ', tf.__version__)
from datasets import load_dataset, Dataset, DatasetDict
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam

from transformers import (AutoTokenizer,
                          create_optimizer,
                          TFAutoModelForTokenClassification,
                         )

#try:
from transformers import DataCollatorForTokenClassification
#except ImportError:
#    from transformers.data.data_collator import default_data_collator as DataCollatorForTokenClassification

#from transformers.keras_callbacks import KerasMetricCallback

import evaluate
from nltk import word_tokenize
import toml
from datetime import datetime as dt
import logging

import sys, os, inspect
#currentdir = os.path.abspath(os.path.curdir)
#parentdir = os.path.dirname(currentdir)
currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,parentdir+'/embed') 

import train_ner as tn
import ner
import ner.llm_utils as llu

from nltk.chunk import conlltags2tree, ChunkScore

#seqeval = evaluate.load('seqeval')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gen_cfg(**kwargs):
    config_path = kwargs.get('configpath', 'config.toml')
    cfg = toml.load(config_path)['finetuning-ner']
    cfg.update(kwargs)
    cfg.update({'wiki_src': 'wikipedia/wiki_definitions_improved.xml.gz',
            'pm_src':  'planetmath/datasets/planetmath_definitions.xml.gz',
            'stacks_src': 'stacks-project/datasets/stacks-definitions.xml.gz',
            'punkt_tok_src': 'trained_models/ner_model/oldie/punkt_params.pickle',
            'glob_data_src': '/train_ner/math*/*.xml.gz',
            'embed_data_src':  'embeddings/model4ner_19-33_02-01/vectors.bin',
            })
    # This is permanent storage# This is permanent storage
    cfg['base_dir'] = os.environ.get('PERMSTORAGE', '/media/hd1') 
    cfg['local_dir'] = os.environ.get('TEMPFASTSTORAGE',
            '/tmp/rm_me_NER')  # This is temporary fast storage

    hoy = dt.now()
    timestamp = hoy.strftime("%b-%d_%H-%M")
    cfg['save_path_dir'] = os.path.join(cfg['local_dir'],
            'trained_ner/lstm_ner/ner_' + timestamp)
    os.makedirs(cfg['save_path_dir'], exist_ok=True)

    FHandler = logging.FileHandler(cfg['local_dir']+"/training.log")
    logger.addHandler(FHandler)

    return cfg

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

def prepare_ds(cfg, model, tokenizer):
    #cfg = tn.gen_cfg()
    text_lst = tn.get_wiki_pm_stacks_data(cfg)
    logger.info(f"The length of text_lst is {len(text_lst)}")
    sent_tok, trainer_params = tn.gen_sent_tokzer(text_lst, cfg)
    tokens_lst, ner_tags_lst, title_lst = ner.bio_tag.put_ner_tags(text_lst, sent_tok)
    def_lst = ner.bio_tag.put_pos_ner_tags(text_lst, sent_tok)

    pos_lst = [[d[0][1] for d in tree_lst['ner']] for tree_lst in def_lst]
    data_dict = {
        'id': list(range(len(tokens_lst))),
        'tokens': tokens_lst,
        'ner_tags': ner_tags_lst,
        'title': title_lst,
        'pos': pos_lst,
    }
    ds = Dataset.from_dict(data_dict)
    temp1_dd = ds.train_test_split(test_size=0.1, shuffle=True)
    temp2_dd = temp1_dd['train'].train_test_split(test_size=0.1, shuffle=True)

    ds = DatasetDict({
        'train': temp2_dd['train'],
        'test': temp1_dd['test'],
        'valid': temp2_dd['test'],
            })

    logger.info(f"{ds=}")
    tokenized_ds = ds.map(tokenize_and_align_labels, batched=True,
            fn_kwargs={'tokenizer': tokenizer})

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
                     return_tensors="tf")

#    tf_train_set = tokenized_ds['train'].to_tf_dataset(
#           columns=['attention_mask', 'input_ids', 'token_type_ids'],
#           label_cols=['token_type_ids'],
#           shuffle=True,
#           collate_fn=data_collator,
#           batch_size=8 )
#
#    tf_valid_set = tokenized_ds['valid'].to_tf_dataset(
#           columns=['attention_mask', 'input_ids', 'token_type_ids'],
#           label_cols=['ner_tags'],
#           shuffle=True,
#           collate_fn=data_collator,
#           batch_size=8 )
#
#    tf_test_set = tokenized_ds['test'].to_tf_dataset(
#           columns=['attention_mask', 'input_ids', 'token_type_ids'],
#           label_cols=['ner_tags'],
#           shuffle=False,
#           collate_fn=data_collator,
#           batch_size=8 )
#

    tf_train_set = model.prepare_tf_dataset(
        tokenized_ds["train"],
        shuffle=True,
        batch_size=cfg['batch_size'],
        collate_fn=data_collator,
    )

    tf_valid_set = model.prepare_tf_dataset(
        tokenized_ds["valid"],
        shuffle=False,
        batch_size=cfg['batch_size'],
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_ds["test"],
        shuffle=False,
        batch_size=cfg['batch_size'],
        collate_fn=data_collator,
    )


    return (tf_train_set, 
            tf_valid_set,
            tf_test_set,
            ds,
           )

def tokenize_and_align_labels(examples, tokenizer):
    'Should be run with the dataset.map method'
    tokenized_inputs = tokenizer(examples['tokens'],
                                truncation=True,
                                is_split_into_words=True)
    
    labels=[]
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def compute_metrics(p):
    '''
    module evaluate was not installed in the initial Singularity container
    '''
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = ['O', 'B-DFNDUM', 'I-DFNDUM']
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    #results = seqeval.compute(predictions=true_predictions, references=true_labels)
    
    return {
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1': results['overall_f1'],
        'accuracy': results['overall_accuracy'],
    }


def compute_chunkscore(ds_test, model, tokenizer, cfg):
    '''
    ds_test: ds['test'] 
    
    '''
    chunkscore = ChunkScore()

    spec_toks = list(tokenizer.special_tokens_map.values())
    spec_toks.remove('[UNK]')


    dif_len_lst = []
    for j in range(len(ds_test)):
        tt = tokenizer(ds_test[j]['tokens'], 
                       return_tensors='tf', is_split_into_words=True)
        logits = model(**tt).logits

        # Grouping entities
        predicted_ids = tf.math.argmax(logits, axis=-1)[0]
        predictions = predicted_ids.numpy().tolist()
        pp = [model.config.id2label[t] for t in predictions]

        wl, il = llu.get_words_back(tt.tokens(),
                              preds=pp, special_tokens=spec_toks)
        try:
            wl, il = llu.join_by_example(wl, ds_test[j]['tokens'], preds=il)
        except AssertionError:
            print(f'Index {j=} caused the error')

        tree_pred = conlltags2tree([(tok, 'Upa', pred) for tok,
                                     pred in zip(wl, il)])

        jdict = ds_test[j]
        bio_tagged = tn.tf_bio_tagger(jdict['ner_tags'])
        tree_gold = conlltags2tree([(jdict['tokens'][i], 
                                     'Upa', 
                                     bio_tagged[i])
                                    for i in range(len(jdict['tokens']))])

        chunkscore.score(tree_pred, tree_gold)
        
        if len(wl) != len(jdict['tokens']):
            dif_len_lst.append(j)
    return chunkscore


def main():
    # get tokenizer

    id2label = {
         0: "O",
         1: "B-DFNDUM",
         2: "I-DFNDUM",
     }
    label2id = {
         "O": 0,
         "B-DFNDUM": 1,
         "I-DFNDUM": 2,
     }

    args = parse_args()
    cfg = gen_cfg(**args)

    xla_gpu_lst = tf.config.list_physical_devices("XLA_GPU")
    logger.info(f'List of XLA GPUs: {xla_gpu_lst}')

    tokenizer = AutoTokenizer.from_pretrained(cfg['checkpoint'])
    
    model = TFAutoModelForTokenClassification.from_pretrained(
        cfg['checkpoint'],
        num_labels=cfg['num_labels'],
        id2label=id2label,
        label2id=label2id,
    )
    
    (tf_train_data, 
    tf_validation_data,
    tf_test_data,
    ds
    ) = prepare_ds(cfg, model, tokenizer)

    cfg['num_train_steps'] = (len(ds['train']) // cfg['batch_size'] *
            cfg['epochs'])

    print(f"####### {cfg['num_train_steps']=}")
    optimizer, lr_schedule = create_optimizer(
        init_lr = 2e-5,  #cfg['init_lr'],
        num_train_steps=cfg['num_train_steps'],
        weight_decay_rate=0.01,   #cfg['weight_decay_rate'],
        num_warmup_steps=10,        #cfg['num_warmup_steps'],
    )

    lr_schedule = PolynomialDecay(
            initial_learning_rate=5e-5, 
            end_learning_rate=0.0, 
            decay_steps=cfg['num_train_steps'],
                )

    #opt = Adam(learning_rate=lr_schedule)
    opt = Adam(learning_rate=2.5e-5)

        
    model.compile(optimizer=opt)

    #metric_callback = KerasMetricCallback(metric_fn=compute_metrics,
    #        eval_dataset=tf_validation_data)
    #callbacks = [metric_callback,]
    model.fit(x=tf_train_data,
              validation_data=tf_validation_data,
              epochs=cfg['epochs'],)
              #callbacks=callbacks)

    chscore = compute_chunkscore(ds['test'], model, tokenizer, cfg)
    print(chscore)
    logger.info(chscore)
    logger.info(f"{cfg=}")

if __name__ == "__main__":
    main()
