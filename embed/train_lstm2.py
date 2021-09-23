from train_lstm import *

def main():
    '''
    Try different decays in the scheduler callbacks
    '''
    args = argParse()
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
    lr = 0.001
    cfg['callbacks'] = ['epoch_times', 'ls_schedule', 'early_stop',]

    og_save_path_dir = cfg['save_path_dir']
    for num, decay in enumerate(np.linspace(0.4, 0.8, args.experiments)):
        cfg['AdamCfg'] = { 'lr': lr, 'lr_decay': decay,}
        cfg['save_path_dir'] = os.path.join(og_save_path_dir , 'exp_{0:0>3}'.format(num))
        os.makedirs(cfg['save_path_dir'], exist_ok=True)

        model = lstm_model_one_layer(embed_matrix, cfg)

        calls, ep_times = model_callbacks(cfg)
        ### FIT THE MODEL ###
        history = model.fit(train_seq, np.array(training[1]),
                        epochs=cfg['epochs'], validation_data=(validation_seq, np.array(validation[1])),
                        batch_size=512,
                        verbose=1,
                        callbacks=calls)
        ## add epoch training times to the history dict
        history.history['epoch_times'] = [t.seconds for t in ep_times.times]
        ## change from np.float32 to float for JSON conversion
        history.history['lr'] = [float(l) for l in history.history['lr']]

        cfg = cutoff_predict_metrics(model, validation_seq, validation, test_seq, test, cfg)

        #save_weights_tokens(model, idx2tkn, history, cfg, subdir='exp_{0:0>3}'.format(num))
        save_tokens_model(model, idx2tkn, history, cfg)


if __name__ == '__main__':
    main()
