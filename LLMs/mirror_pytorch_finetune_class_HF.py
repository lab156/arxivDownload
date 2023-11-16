import io
import sys
import os
import torch
import glob
import numpy as np
import toml
import json
#from tqdm.notebook import tqdm
from tqdm import tqdm
from torch.utils.data import  DataLoader

# Imports for distributed training
import torch.distributed as dist
#from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

#from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
import sklearn.metrics as metrics
from lxml import etree
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification,
                          AutoTokenizer,
                          AutoModelForSequenceClassification,
                          MistralConfig,
                          LlamaTokenizer,
                          MistralForSequenceClassification,
                          TrainingArguments,
                          Trainer,)

from datasets import Dataset, DatasetDict, load_metric
from datetime import datetime as dt

currentdir = os.path.abspath(os.path.curdir)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,currentdir) 
sys.path.insert(0,currentdir+'/embed') 
print(f"{currentdir}/embed")
from classifier_trainer.trainer import stream_arxiv_paragraphs
import parsing_xml as px
import peep_tar as peep

#from train_lstm import gen_cfg, find_best_cutoff
from extract import Definiendum

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dist.init_process_group("nccl")
# rank variable is global
rank = dist.get_rank()
print(f"Start running basic DDP example on rank {rank}.\n")
logger.info(f"Start running basic DDP example on rank {rank}.\n")

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
    cfg['n_workers'] = cfg['n_workers']
    
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

class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask. 
    
    It uses a given tokenizer and label encoder to convert any text and labels 
    to numbers that 
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length \
               if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder

        return
    
    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this 
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        labels = [sequence['label'] for sequence in sequences]
        # Encode all labels using label encoder.
        #labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with 
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt",
                                    padding=True, truncation=True,
                                    max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})

        return inputs
    
def from_pretrained_model_and_tokenizer(device, cfg):
    # Get model configuration.
    #model_config = GPT2Config.from_pretrained(
    model_config = MistralConfig.from_pretrained(
                           pretrained_model_name_or_path=cfg['checkpoint'],
                                              num_labels=2)

    # Get model's tokenizer.
    print('Loading tokenizer...')
    #tokenizer = GPT2Tokenizer.from_pretrained(
    tokenizer = LlamaTokenizer.from_pretrained(
                           pretrained_model_name_or_path=cfg['checkpoint'])
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token


    # Get the actual model.
    print('Loading model...')
    #model = GPT2ForSequenceClassification.from_pretrained(
    model = MistralForSequenceClassification.from_pretrained(
                           pretrained_model_name_or_path=cfg['checkpoint'],
                            config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Load model to defined device.
    #model.to(device)
    #print('Model loaded to `%s`'%device)
    #logger.info('Model loaded to `%s`'%device)
 
    return model, tokenizer

def prepare_dataloaders(ds, tokenizer, cfg):
    ds = ds.select(range(int(cfg['shrink_data_factor']*len(ds))))
    temp1_ds = ds.train_test_split(test_size=0.1, shuffle=True)
    temp2_ds = temp1_ds['train'].train_test_split(test_size=0.1, shuffle=True)

    ds = DatasetDict({
        'train': temp2_ds['train'],
        'test': temp1_ds['test'],
        'valid': temp2_ds['test'],
    })
    
    cfg['labels_ids'] = {'neg': 0, 'pos': 1}

    gpt2_classification_collator = Gpt2ClassificationCollator(
            use_tokenizer=tokenizer, 
            labels_encoder=cfg['labels_ids'], 
          max_sequence_len=cfg['max_length'])

    # Create pytorch dataset.

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(ds['train'],
                                  batch_size=cfg['batch_size'], 
                                  shuffle=True, 
                                  collate_fn=gpt2_classification_collator)
    logger.info('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    # Create pytorch dataset.
    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(ds['valid'], 
                                  batch_size=cfg['batch_size'], 
                                  shuffle=False, 
                                  collate_fn=gpt2_classification_collator)
    logger.info('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

    test_dataloader = DataLoader(ds['test'], 
                                  batch_size=cfg['batch_size'], 
                                  shuffle=False, 
                                  collate_fn=gpt2_classification_collator)
    logger.info('Created `test_dataloader` with %d batches!'%len(valid_dataloader))

    return (train_dataloader,
            valid_dataloader,
            test_dataloader,
            cfg,
            )

def train(dataloader, model, optimizer_, scheduler_, device_):
    r"""
  Train pytorch model on a single pass through the data loader.

  It will use the global variable `model` which is the transformer model 
  loaded on `_device` that we want to train on.

  This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

  Arguments:

      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.

      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.

      device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.

  Returns:

      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss].
  """

  # Tracking variables.
    predictions_labels = []
    true_labels = []
  # Total loss for this epoch.
    total_loss = 0

  # Put the model into training mode.
    model.train()

  # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):

    # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this a bert model function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer_.step()

        # Update the learning rate.
        scheduler_.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

  # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)
  
  # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
 
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, 
           references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

def train_model(model, dataset, tokenizer, collator, cfg):
    # Taken from https://docs.databricks.com/en/_extras/notebooks/source/deep-learning/distributed-fine-tuning-hugging-face.html
 
    training_args = TrainingArguments(
      output_dir=cfg['savedir'],
      learning_rate=cfg['initial_lr'],
      per_device_train_batch_size=cfg['batch_size'],
      per_device_eval_batch_size=cfg['batch_size'],
      num_train_epochs=cfg['num_epochs'],
      weight_decay=cfg['weight_decay'],
      save_strategy="epoch",
      report_to=[], # REMOVE MLFLOW INTEGRATION FOR NOW
      push_to_hub=False,  # DO NOT PUSH TO MODEL HUB FOR NOW,
      load_best_model_at_end=True, # RECOMMENDED
      metric_for_best_model="eval_loss", # RECOMMENDED
      evaluation_strategy="epoch" # RECOMMENDED
    )
 
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_train,
      eval_dataset=tokenized_test,
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.state.best_model_checkpoint


def validation(dataloader, model, device_):
    r"""Validation function to evaluate model performance on a 
  separate set of data.

  This function will return the true and predicted labels so we can use later
  to evaluate the model's performance.

  This function is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed 
    straight into the model - `model(**batch)`.

  Arguments:

    dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.

    device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.

  Returns:
    
    :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss]
  """

  # Tracking variables
    predictions_labels = []
    true_labels = []
  #total loss for this epoch.
    total_loss = 0

  # Put the model in evaluation mode--the dropout layers behave differently
  # during evaluation.
    model.eval()

  # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):

        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():        

        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        # token_type_ids is the same as the "segment ids", which 
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple along with the logits. We will use logits
        # later to to calculate training accuracy.
            loss, logits = outputs[:2]
        
        # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
            total_loss += loss.item()
        
        # get predicitons to list
            predict_content = logits.argmax(axis=-1).flatten().tolist()

        # update list
            predictions_labels += predict_content

  # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

  # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss

def main():
    args = parse_args()
    xml_lst, cfg = gen_cfg(**args)
    assert len(xml_lst) > 0, 'Empty xml_lst'

    if rank == 0:
        print('###### Number of workers=', cfg['n_workers'])
        num_gpus = torch.cuda.device_count()
        print(f'###### Number of GPUs={num_gpus}')
        logger.info(f"###### Number of workers={cfg['n_workers']}")
        logger.info(f'###### Number of GPUs={num_gpus}')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #logger.info(f"{device=}")
    device = None

    model, tokenizer = from_pretrained_model_and_tokenizer(device, cfg)
    ds = get_dataset(xml_lst, cfg)
    gpt2_classification_collator = Gpt2ClassificationCollator(
            use_tokenizer=tokenizer, 
            labels_encoder=cfg['labels_ids'], 
          max_sequence_len=cfg['max_length'])
    trainer_best = train_model(model, ds, tokenizer, 
            gpt2_classification_collator, cfg)

    print(f"Esto es {trainer_best=}")


    #tokenizer = AutoTokenizer.\
    #        from_pretrained(cfg['checkpoint'], truncation=True)
    #model = AutoModelForSequenceClassification.\
    #        from_pretrained(cfg['checkpoint'], num_labels=2)
#    model.to(device)
#
#    (train_dataloader,
#    valid_dataloader,
#    test_dataloader,
#    cfg) = prepare_dataloaders(ds, tokenizer, cfg)
#
#    optimizer = AdamW(model.parameters(),
#                      lr = cfg['initial_lr'],
#                      eps = 1e-8 # default is 1e-8.
#                      )
#
#    # Total number of training steps is number of batches * number of epochs.
#    # `train_dataloader` contains batched data so `len(train_dataloader)` gives 
#    # us the number of batches.
#    total_steps = len(train_dataloader) * cfg['num_epochs']
#
#    # Create the learning rate scheduler.
#    scheduler = get_linear_schedule_with_warmup(optimizer, 
#                            num_warmup_steps = 0, # Default value in run_glue.py
#                            num_training_steps = total_steps)
#
#    # Store the average loss after each epoch so we can plot them.
#    train_stats = {'train_loss':[], 'val_loss':[],
#                   'train_acc':[], 'val_acc':[]}
#
#    # Loop through each epoch.
#    print('Epoch')
#    worse_in_a_row = 0
#    val_acc_best = 0
#    for epoch in range(cfg['num_epochs']):
#        print(f"##### EPOCH {epoch+1} / {cfg['num_epochs']} ####")
#        print('Training on batches...')
#        # Perform one full pass over the training set.
#        train_labels, train_predict, train_loss = train(train_dataloader, 
#                model,
#                optimizer, scheduler, device)
#        train_acc = accuracy_score(train_labels, train_predict)
#
#        # Get prediction form model on validation data. 
#        print('Validation on batches...')
#        valid_labels, valid_predict, val_loss = validation(valid_dataloader, 
#                model, device)
#        val_acc = accuracy_score(valid_labels, valid_predict)
#
#        # Print loss and accuracy values to see how training evolves.
#        logger.info("epoch: %i - train_loss: %.5f - val_loss: %.5f \
#- train_acc: %.5f - valid_acc: %.5f"\
#            %(epoch, train_loss, val_loss, train_acc, val_acc))
#        print("train_loss: %.5f - val_loss: %.5f - train_acc: %.5f \
#- valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
#
#        # Store the loss value for plotting the learning curve.
#        train_stats['train_loss'].append(train_loss)
#        train_stats['val_loss'].append(val_loss)
#        train_stats['train_acc'].append(train_acc)
#        train_stats['val_acc'].append(val_acc)
#
#        if val_acc_best < val_acc:
#            # Improved
#            worse_in_a_row = 0
#            print(f"Saving to {cfg['savedir']}")
#            print()
#            model.save_pretrained(save_directory=cfg['savedir'])
#            tokenizer.save_pretrained(save_directory=cfg['savedir'])
#            with open(os.path.join(cfg['savedir'], 'train_stats.json'), 'w') as fobj:
#                json.dump(train_stats, fobj)
#            val_acc_best = val_acc
#        else:
#            worse_in_a_row += 1
#
#        if worse_in_a_row >= 2:
#            break
#
#
#    preds = validation(test_dataloader, model, device)
#    metric_str = metrics.classification_report(preds[0], preds[1])
#    f1_score = metrics.f1_score(preds[0], preds[1])
#    print(metric_str)
#    print(f"{f1_score=}")
#    logger.info(f"{f1_score=}")
#    cfg['f1_score'] = f1_score
#    logger.info(metric_str)
#
#    #Save the model
#    if cfg['savedir'] != '':
#        pass
#    else:
#        logger.warning(
#        "cfg['savedir'] is empty string, not saving model.")
#    logger.info(cfg)

if __name__ == "__main__":
    main()
