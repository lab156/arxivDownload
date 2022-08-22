import sys
import argparse
from clean_and_token_text import normalize_text

def deal_with_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('embed4', choices=['embed4classif', 'embed4ner'],
            help='choose between classification (removes all punct) or ner cleaning')
    parser.add_argument('--in_files', type=str, default=None,
            help='read a file to clean  e.g. infile.txt')
    parser.add_argument('--out_file', type=str, default=None,
            help='file name to output the clean text e.g. outfile.txt')
    #parser.add_argument('--skip_n', default=1, type=int)
    args = parser.parse_args()
    return args


def main1():
    if sys.stdin.isatty():
        print('not piped me')
    else:
        # this is the case where we need to read
        text = sys.stdin.read()

    args = deal_with_args()
    if args.embed4 == 'embed4classif':
         clean_text =  normalize_text(text, 'rm_punct')
    elif args.embed4 == 'embed4ner': 
         clean_text =  normalize_text(text)
    else:
        raise NotImplementedError('I dont know what the embed4 value means')
    print(clean_text)

if __name__ == '__main__':
    main1()


