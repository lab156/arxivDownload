from train_lstm import *

def argParse3():
    '''
    Parsing all arguments 
    specific for this script 
    '''
    import argparse
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument('--epochs', type=int, default=2,
            help="Number of epochs to train. Overrides default value")
    parser.add_argument('--experiments', type=int, default=2,
            help="Number of experiment loops to do.")
    parser.add_argument('--cells', type=int, default=0,
            help="Number of first layer LSTM cells.")
    parser.add_argument('--startat', type=int, default=0,
            help="Start the experiments at 64*value. Heads up with the 64.")
    parser.add_argument('-p', '--profiling', action='store_true',
            help="Set the profiling mode to True (default False)")
    parser.add_argument('-m', '--mini', action='store_true',
            help="Set a small version of the training data set.")
    args = parser.parse_args()
    return args

def main():
    '''
    Increase the size of the neural network by changing the number of LSTM
    Cells

    Options:
    --experiments: Number of sizes in steps of 64 to try.
    -m: minified version of the data set.

wembed_basename = 'embeddings/glove_model_18-31_15-08'
    '''
    args = argParse3()
    xml_lst, cfg = gen_cfg(parsed_args = args)

    logging.basicConfig(filename=os.path.join(cfg['save_path_dir'], 'training.log'),
            level=logging.INFO)
    logger.info("GPU devices: {}".format(list_physical_devices('GPU')))
    logger.info("Length of the xml_lst is: {}".format(len(xml_lst)))

    train_seq, validation_seq, test_seq,\
    idx2tkn, tkn2idx, training, validation,\
    test, cfg = read_train_data(xml_lst, cfg)

    embed_matrix, cfg = gen_embed_matrix(tkn2idx, cfg)

    #### FIT LOOP ####
    cfg['callbacks'] = ['epoch_times', 'ls_schedule', 'early_stop',]
    cfg['AdamCfg'] = { 'lr': 0.001, 'lr_decay': 0.5,}

    og_save_path_dir = cfg['save_path_dir']
#    for num, decay in enumerate(np.linspace(0.4, 0.8, args.experiments)):
    for num in range(args.experiments):
        cells = 64*(num + 1) + 64*int(args.startat)
        cfg['lstm_cells'] = cells
        logger.info("\n Starting Experiment {} -- training with Number of Cells: {} \n"\
                        .format(num + 1 , cells))

        cfg['save_path_dir'] = os.path.join(og_save_path_dir , 'exp_{0:0>3}'.format(num + 1))
        os.makedirs(cfg['save_path_dir'], exist_ok=True)

        model = lstm_model_one_layer(embed_matrix, cfg)

        calls, ep_times = model_callbacks(cfg)
        ### FIT THE MODEL ###
        history = model.fit(train_seq, np.array(training[1]),
                        epochs=cfg['epochs'], validation_data=(validation_seq,
                                              np.array(validation[1])),
                        batch_size=512,
                        verbose=1,
                        callbacks=calls)
        ## add epoch training times to the history dict
        history.history['epoch_times'] = [t.seconds for t in ep_times.times]
        ## change from np.float32 to float for JSON conversion
        history.history['lr'] = [float(l) for l in history.history['lr']]

        cfg = cutoff_predict_metrics(model, validation_seq, validation,
                test_seq, test, cfg)

        #save_weights_tokens(model, idx2tkn, history, cfg, subdir='exp_{0:0>3}'.format(num))
        save_tokens_model(model, idx2tkn, history, cfg)


if __name__ == '__main__':
    main()
