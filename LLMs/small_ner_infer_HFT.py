import re
import os
from os.path import join


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
    args = parser.parse_args()

    # make sure --savepath exists
    #if args.savedir != '':
    #    os.makedirs(args.savedir, exist_ok=True)

    return vars(args)

reg_expr = re.compile('"(.+?)"+')
def get_term(out_str):
    # The out_str is the output string produced by the LLM
    Defin = reg_expr.findall(out_str)
    return Defin

reg_expr2 = re.compile('Definition\s+[\d\.]+\s+(.+)')
def get_text(in_str):
    Defin = reg_expr2.findall(in_str)
    return Defin

def main():
    args = parse_args()
    
    cfg = {'checkpoint': 'bert-base-uncased',
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
        
    print(get_text(xdefs_in_lst[15]) )

if __name__ == "__main__":
    main()

    