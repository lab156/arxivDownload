from train_ner import *

def gen_cfg_wrap():
    args = argParse()
    cfg = gen_cfg(args = args)

    cfg['epochs'] = args.epochs
    cfg['n_experiments'] = args.experiments
    cfg['profiling'] = False

    
    return cfg

def main():
    '''
    Try different decays in the scheduler callbacks
    '''
    cfg = gen_cfg_wrap()

    logging.basicConfig(filename=os.path.join(cfg['save_path_dir'], 'training.log'),
            level=logging.INFO)
    logger.info("GPU devices: {}".format(list_physical_devices('GPU')))

    # RETRIEVE AND PREP DATA -----------------------------------------------
    text_lst = get_wiki_pm_stacks_data(cfg)
    sent_tok, trainer_params = gen_sent_tokzer(text_lst, cfg)
    logger.info(sent_tok._params.abbrev_types)

    def_lst = ner.bio_tag.put_pos_ner_tags(text_lst, sent_tok)
    random.shuffle(def_lst)
    logger.info("Length of the def_lst is: {}".format(len(def_lst)))

    pos_ind_dict, pos_lst, cfg = get_pos_ind_dict(def_lst, cfg)

    wind, embed_matrix, cfg = open_word_embedding(cfg)

    train_def_lst, test_def_lst, valid_def_lst = get_ranges_lst(def_lst, cfg)

    train_seq, train_pos_seq, train_bin_seq , train_lab = prep_data4real(
            train_def_lst, wind, pos_ind_dict, cfg)
    test_seq, test_pos_seq, test_bin_seq , test_lab = prep_data4real(
            test_def_lst, wind, pos_ind_dict, cfg)
    valid_seq, valid_pos_seq, valid_bin_seq , valid_lab = prep_data4real(
            valid_def_lst, wind, pos_ind_dict, cfg)

    #### TRAIN LOOP ####
    #cfg['callbacks'] = ['epoch_times', 'ls_schedule', 'early_stop',]

    og_save_path_dir = cfg['save_path_dir']
    for i in range(cfg['n_experiments']):
        for j in range(i + 1):
            cfg['lstm_units1'] = 50*(i + 3) # Start from 150,150
            cfg['lstm_units2'] = 50*(j + 3)
            num = cfg['n_experiments']*i + j
            cfg['save_path_dir'] = os.path.join(og_save_path_dir , 'exp_{0:0>3}'.format(num))
            os.makedirs(cfg['save_path_dir'], exist_ok=True)

            cfg['callbacks'] = ['epoch_times', 'early_stop']
            calls, ep_times = model_callbacks(cfg)
            ### FIT THE MODEL ###
            # TRAIN AND DEFINE MODEL ---------------------------------------------
            model_bilstm = bilstm_model_w_pos(embed_matrix, cfg)
            #history = train_model(train_seq, train_lab, test_seq, test_lab, model_bilstm_lstm, cfg )
            history = model_bilstm.fit([train_seq, train_pos_seq, train_bin_seq], train_lab, 
                    epochs=cfg['epochs'],
                    batch_size=cfg['batch_size'],
                    validation_data=([valid_seq, valid_pos_seq, valid_bin_seq], valid_lab),
                    callbacks=calls)

            boy_f, cfg = tboy_finder(model_bilstm, test_seq, 
                    test_pos_seq, test_bin_seq, test_lab, test_def_lst, cfg)

            # SAVING DATA AND SOMETIMES MODELS ---------------------------------------
            save_cfg_data(model_bilstm, cfg, wind, pos_lst, trainer_params)


if __name__ == '__main__':
    main()
