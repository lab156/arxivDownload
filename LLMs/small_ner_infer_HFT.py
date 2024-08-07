import re
import os
from os.path import join
import numpy as np

import tensorflow as tf
from transformers import (AutoTokenizer,
                          create_optimizer,
                          TFAutoModelForTokenClassification,
                          pipeline,
                         )

from transformers import DataCollatorForTokenClassification


reg_expr = re.compile('"(.+?)"+')
def get_term(out_str):
    # The out_str is the output string produced by the LLM
    Defin = reg_expr.findall(out_str)
    return Defin

reg_expr2 = re.compile('Definition\s+[\d\.]+\s+(.+)')
def get_text(in_str):
    Defin = reg_expr2.findall(in_str)
    return Defin


def sanity_check(model, tokenizer, text=None):
    if text == None:
        text = """We define a Banach space as a complete vector normed space."""
    #text = ''
    #j = 19
    #for t in ds['test'][j]['tokens']:
    #    text += t + ' '
    print(f'{text=}')
    classifier = pipeline('ner', model=model, tokenizer=tokenizer)
    print('The pipeline result is: ', classifier(text))

    inputs = tokenizer(text, return_tensors='tf')
    logits = model(**inputs).logits
    predicted_ids = tf.math.argmax(logits, axis=-1)
    predicted_token_class = [model.config.id2label[t] 
                 for t in predicted_ids[0].numpy().tolist()]

    for i in range(len(predicted_token_class)):
        print(inputs.tokens()[i], predicted_token_class[i])

    #tt = tokenizer(ds['test'][j]['tokens'], return_tensors='tf', is_split_into_words=True)
    tt = tokenizer(text, return_tensors='tf', is_split_into_words=False)
    logits = model(**tt).logits
    print(f"{logits=}")

    # Grouping entities
    predicted_ids = tf.math.argmax(logits, axis=-1)[0]
    predictions = predicted_ids.numpy().tolist()
    results = []
    inputs_with_offsets = tokenizer(text, return_offsets_mapping=True)
    tokens = inputs_with_offsets.tokens()
    offsets = inputs_with_offsets['offset_mapping']

    probs = tf.math.softmax(logits, axis=-1)[0]
    probs = probs.numpy().tolist()

    #start, end = inputs.word_to_chars(10)
    end = 0

    idx = 0
    while idx < len(predictions):
        pred = predictions[idx]
        label = model.config.id2label[pred]
        if label != 'O':
            label = label[2:]
            start, end = offsets[idx] # 2nd output is the end of word
            #idx += 1
            
            # Grab all tokens labeled with an I-label
            all_scores = []
            while (
                idx < len(predictions)
                and model.config.id2label[predictions[idx]][2:] == label
                   ):
                all_scores.append(probs[idx][pred])
                _, end = offsets[idx]
                idx += 1
                
            score = np.mean(all_scores).item()
            word = text[start:end]
            results.append(
                {'entity': label, 
                 'score': score,
                 'word': word,
                'start': start,
                'end': end,}
            )
        idx += 1
    print(results)

def parse_args():
    '''
    parse args should be run before gen_cfg
    '''
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--savedir', type=str, default='',
    #    help="""Path to save the finetuned model, dir name only.""")
    parser.add_argument('--xdefs', type=str, default='',
        help="""path to the JUrban/extract-defs cloned repository""")
    parser.add_argument('--model', type=str,
            default='/home/luis/ner_model',
            help='Path to the tensorflow model directory')
    args = parser.parse_args()

    # make sure --savepath exists
    #if args.savedir != '':
    #    os.makedirs(args.savedir, exist_ok=True)

    return vars(args)

def main():
    args = parse_args()
    #print(f'{args=}')
    
    cfg = {'checkpoint': 'bert-large-cased',
      'max_length': 150, # check mp_infer_HFTrans_ner.py
      }

    xdefs_root = args['xdefs']
    xdefs_inputs = join(xdefs_root, 'lm-inputs/defsCT')
    xdefs_outputs = join(xdefs_root, 'lm-outputs/defsCT')
    xdefs_inputs_filelst = sorted(os.listdir(xdefs_inputs))
    xdefs_outputs_filelst = sorted(os.listdir(xdefs_outputs))
    
    with open(join(xdefs_outputs, xdefs_outputs_filelst[0]), 'r') as fobj:
        xdefs_out_lst = fobj.readlines()
    with open(join(xdefs_inputs, xdefs_inputs_filelst[0]), 'r') as fobj:
        xdefs_in_lst = fobj.readlines()

    # LLM loading and preparation
    tokenizer = AutoTokenizer\
            .from_pretrained(cfg['checkpoint'])
        
    print(get_text(xdefs_in_lst[15]) )
    tf_model_dir = args['model']
    cfg = {#'outdir': args['out'],
            'max_length': 150, 
            'inference_batch_size': 250}
    #with open(os.path.join(tf_model_dir, 'config.json')) as fobj:
    #    cfg['checkpoint'] = json.loads(fobj.read())['_name_or_path']
    Model = TFAutoModelForTokenClassification\
            .from_pretrained(tf_model_dir)
    sanity_check(Model, tokenizer, text=get_text(xdefs_in_lst[5]))

if __name__ == "__main__":
    main()

    