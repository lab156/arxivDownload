import functools
from collections import defaultdict, Counter
import re
import unidecode
from lxml import etree
import glob
from tqdm import tqdm
import multiprocessing as mp
import os


def normalize_text(text, *vargs, **kwargs):
    '''
    a copy of the normalize_text function in the word2vec repo
    see the `demo-train-big-model.sh` file
    run tests with python3 -m doctest -v embed_utils.py
    Options:
        # rm_punct
        # rm_special_chars

    >>> normalize_text('hi, there.')
    'hi , there . '

    >>> normalize_text('This |is\t \twork=ing')
    'this is work ing'

    >>> normalize_text('remove the <br> <br /> <br     /> and not the <')
    'remove the and not the <'

    >>> normalize_text('en 1492, Colon llegó a ?')
    'en , colon llego a ? '

    >>> normalize_text('I rem/ember káhler painlevé in § 74', 'rm_special_chars')
    'i rem / ember khler painlev in '

    >>> normalize_text('Kähler manifolds are, fun, and interesting.')
    'kahler manifolds are , fun , and interesting . '

    >>> normalize_text('Málaga Kähler  Étale française problème')
    'malaga kahler etale francaise probleme'

    TESTS WITH rm_punct

    >>> normalize_text('en 1492, Colón llegó a ?', 'rm_punct')
    'en colon llego a '

    >>> normalize_text('remove the <br> <br /> <br     /> and the <', 'rm_punct')
    'remove the and the '

    >>> normalize_text('en el siglo XV, Colón llego a ?', 'rm_punct')
    'en el siglo xv colon llego a '

    >>> normalize_text('restricts to a ”weak” symplectic', 'rm_punct')
    'restricts to a weak symplectic'

    >>> normalize_text('a Hamiltonian action, i.e. a Lie algebra', 'rm_punct')
    'a hamiltonian action ie a lie algebra'

    >>> normalize_text('¡al sonoro rugir del cañón!\\n', 'rm_punct')
    'al sonoro rugir del canon \\n '

    >>> normalize_text(' Well-defined and Z-module [?] have  dashes [?]', 'rm_punct')
    ' well defined and z module have dashes '

    '''

    # Intended purpose: NER
    repl_list = [("’","") ,
            ("′","") ,
           ("''", ""),
            ("'","") ,
            ("“",'') ,
            ('"','') ,
            ('.',' . ') ,
            (', ',' , ') ,
            ('; ',' ; ') ,
            (': ',' : ') ,
            ('(',' ( ') ,
            (')',' ) ') ,
            ('[', ''),
            (']', ''),
            ('{', ''),
            ('}', ''),
            ('!',' ! '),
            ('?',' ? ') ,
            ('-',' - ') ,
            ('=',' ') ,
            ('=',' ') ,
            ('*',' ') , 
            ('|',' ') ,
            ('/',' / ') ,
            ('«',' ') ,
            ('»', ' '),
            ('\n', ' \n ')]

    # Intended purpose: general embedding of phrases
    punct_list = [ ("’", ''),
            ("'", ''),
            ("′", ''),
            ("''", ''),
            ("'", ''),
            ("“", ''),
            ('"', ''),
            ('.', ''),
            (',', ''),
            ('(', ''),
            (')', ''),  
            ('!', ''),
            ('?', ''),
            (';', ''),
            (':', ''),
            ('=', ''),
            ('*', ''),
            ('|', ''),
            ('/', ''),
            ('«', ''),
            ('»', ''),
            ('<', ''),
            ('>', ''),
            ('”', ''),
            ('[', ''),
            (']', ''),
            ('{', ''),
            ('}', ''),
            ('-', ' '),
            ('\n', ' \n ')]  # Dashes are replaced by spaces because might be included in phrases
            # for example well-defined and well defined should be in the phrases list

    text = re.sub(r'<br\s*/? ?>', ' ', text) # remove <br /> variants
    text = re.sub(r'[0-9]+', ' ', text)

    if 'rm_special_chars' in vargs:
        text = re.sub(r'[^\x00-\x7F]+', '', text)
    else:
        text = unidecode.unidecode(text)

    if 'rm_punct' in vargs:
        text = functools.reduce(lambda a,b: a.replace(*b), [text] + punct_list)
    else:
        text = functools.reduce(lambda a,b: a.replace(*b), [text] + repl_list)

    text = re.sub(r'[ \t]+', ' ', text) # Normalize all spaces (\s) to one blank space

    return text.lower()

def token_phrases(text, phrase, join_str='_'):
    '''
    expecting clean text

    >>> token_phrases('hi riemann how is your riemann metric today', 'riemann integral')
    'hi riemann how is your riemann metric today '

    >>> token_phrases('hi riemann how is your riemann integral today', 'riemann integral')
    'hi riemann how is your riemann_integral today '

    >>> token_phrases('hi rr how is your rr ii ii today rr ii vv oo', 'rr ii')
    'hi rr how is your rr_ii ii today rr_ii vv oo '

    >>> token_phrases('hi rr how is your rr ii uu today rr ii vv oo', 'rr ii uu')
    'hi rr how is your rr_ii_uu today rr ii vv oo '

    >>> token_phrases('hi rr how is your rr ii uu today rr ii oo uu', 'ii oo uu')
    'hi rr how is your rr ii uu today rr ii_oo_uu '
    '''
    text_lst = text.split()
    ngram_len = len(phrase.split())
    #sub_token = '_'.join(phrase.split())
    mod_text = ''
    k = 0
    while k < len(text_lst) :
        ngram = text_lst[k:(k+ngram_len)]
        if ngram == phrase.split():
            mod_text += join_str.join(phrase.split()) + ' '
            k += ngram_len
        else:
            mod_text += text_lst[k] + ' '
            k += 1

    return mod_text #+ ' '.join(text_lst[:(k - len(text_lst))])

def token_phrases2(text, phrase_lst, join_str='_'):
    '''
    Second more efficient try

    >>> token_phrases2('hi riemann how is your riemann metric today', ['riemann integral'])
    'hi riemann how is your riemann metric today '

    >>> token_phrases2('hi riemann how is your riemann integral today', ['riemann integral'])
    'hi riemann how is your riemann_integral today '

    >>> token_phrases2('hi rr how is your rr ii ii today rr ii vv oo', ['rr ii'])
    'hi rr how is your rr_ii ii today rr_ii vv oo '

    >>> token_phrases2('hi rr how is your rr ii uu today rr ii vv rr', ['rr ii uu'])
    'hi rr how is your rr_ii_uu today rr ii vv rr '

    >>> token_phrases2('hi rr how is your rr ii uu today rr ii oo uu \\n ', ['ii oo uu', 'ii uu'])
    'hi rr how is your rr ii_uu today rr ii_oo_uu \\n  '

    >>> token_phrases2('hi rr how is your rr ii ii today rr ii vv oo', ['rr ii', 'how is'])
    'hi rr how_is your rr_ii ii today rr_ii vv oo '
    '''
    # First prepare the phrases
    phrase_dict = defaultdict(set)
    for ph in phrase_lst:
        ph_lst = ph.strip().split()
        if len(ph_lst) > 1:
            phrase_dict[ph_lst[0]].add(tuple(ph_lst[1:]))
        else:
            raise ValueError('A phrase with only too few words was given. Phrase: {}'.format(ph))

    text_lst = text.split(' ')
    phrase_dict = dict(phrase_dict)

    k = 0
    mod_text = ''
    while k < len(text_lst):
        try:
            next_words_lst = phrase_dict[text_lst[k]]
            advance_just_one_word = True
            for nw in next_words_lst:
                ngram_len = len(nw) + 1
                nw_text = tuple(text_lst[ (k+1) : (k+ngram_len) ])
                if nw == nw_text:
                    mod_text += join_str.join(text_lst[k: k+ngram_len]) + ' '
                    k += ngram_len
                    advance_just_one_word = False
            if advance_just_one_word:
                mod_text += text_lst[k] + ' '
                k += 1
        except KeyError:
            mod_text += text_lst[k] + ' '
            k += 1
        except IndexError:
            import pdb; pdb.set_trace()

    return mod_text


def token_phrases3(text, phrase_lst, join_str='_'):
    '''
    >>> token_phrases3('hi rr how is your rr ii uu today rr ii oo uu \\n ', ['ii oo uu', 'ii uu'])
    'hi rr how is your rr ii_uu today rr ii_oo_uu \\n '

    >>> token_phrases3('hi rr how is your rr ii uu today rr ii oo uu ', ['ii oo uu', 'ii uu'])
    'hi rr how is your rr ii_uu today rr ii_oo_uu '

    >>> token_phrases3('hi rr how is your rr ii ii today rr ii vv oo', ['rr ii'])
    'hi rr how is your rr_ii ii today rr_ii vv oo '

    >>> token_phrases3('hi rr how is your rr ii uu today rr ii vv rr', ['rr ii uu'])
    'hi rr how is your rr_ii_uu today rr ii vv rr'

    >>> token_phrases3('hi rr how is your rr ii ii today rr ii vv oo', ['rr ii', 'how is'])
    'hi rr how_is your rr_ii ii today rr_ii vv oo '

    >>> token_phrases3('rr ii _inline_ _rr_ii_', ['rr ii', 'how is'])
    'rr_ii _inline_ _rr_ii_ '
    '''
    phrase_default = defaultdict(set)
    for ph in phrase_lst:
        ph_lst = ph.strip().split()
        if len(ph_lst) > 1:
            phrase_default[ph_lst[0]].add(tuple(ph_lst[1:]))
        else:
            raise ValueError('A phrase with only too few words was given. Phrase: {}'.format(ph))

    text_lst = text.split(' ')
    phrase_dict = {}
    for k, v in phrase_default.items():
        # phrase_dict values are lists of phrases sorted by their length (shorter first)
        phrase_dict[k] = sorted(v, key=len)
    del phrase_default # clean up the defaultdict

    text_iter = (x.group(0) for x in re.finditer(r"[a-z\n_]+", text))
    mod_text = ''
    words = []
    while True:
        try:
            if words == []:
                words.append(next(text_iter))
            phrase_lst = phrase_dict[words[0]]
            words += [next(text_iter) for _ in phrase_lst[0]]
            advance_just_one_word = True
            for ph in phrase_lst:
                ph_len = len(ph)
                if len(words) <= ph_len:
                    words += [next(text_iter) for _ in range(ph_len - len(words) + 1)] 
                #import pdb; pdb.set_trace()
                if tuple(words[1: (ph_len+1) ]) == ph:
                    mod_text +=  join_str.join([words.pop(0) for _ in range(ph_len+1)]) + ' '
                    advance_just_one_word = False
                    break
            if advance_just_one_word:
                mod_text += words.pop(0) + ' '
        except KeyError:
            mod_text += words.pop(0) + ' '
            continue
        except StopIteration:
            break
    return mod_text + (( ' '.join(words)) if words != [] else '')




def next_word_dict(phrases):
    """
    takes a list of phrases
    makes a dict with word: [list of next words]

   # >>> next_word_dict(['ri me'])
   # defaultdict(<class 'set'>, {'ri': {'me'}})

   # >>> next_word_dict(['ri me', 'ri in'])
   # defaultdict(<class 'set'>, {'ri': {'me', 'in'}})

   # >>> next_word_dict(['ri me', 'ri in', 'in la'])
   # defaultdict(<class 'set'>, {'ri': {'me', 'in'}, 'in': {'la'}})

   # >>> next_word_dict(['do re mi', 'mi fa sol', 'fa sol la'])
   # defaultdict(<class 'set'>, {'do': {'re'}, 're': {'mi'}, 'mi': {'fa'}, 'fa': {'sol'}, 'sol': {'la'}})
    """
    # Create the Next Word Dictionary (nwd)
    nwd = defaultdict(set)

    for ph in phrases:
        ph_lst = ph.strip().split()
        if len(ph_lst) > 1:
            for k, tok in enumerate(ph_lst[:-1]):
                nwd[tok].add(ph_lst[k + 1])
    return nwd

def tokenize_and_write(in_file, out_file, phrases_list):
    in_fobj = open(in_file, 'r') 
    out_fobj = open(out_file, 'a') 
    print("Writing {} to {}".format(in_fobj.name, out_fobj.name))
    while (line := in_fobj.readline()) != '':
        line = normalize_text(line)
        #line = token_phrases3(line, phrases_list)
        out_fobj.write(line)
    in_fobj.close()
    out_fobj.close()

phrase_blacklist = ['_inline_math_ and',
        '_inline_math_ _inline_math_',
        'recent years',
        '_inline_math_ of',
        'for _inline_math_',
        '_inline_math_ if',
        '_inline_math_ also',
        'suppose that',
        'condition for',
        '_inline_math_ on',
        '_inline_math_ a',
        'family of',]


if __name__ == "__main__":
    '''
    Examples:
    time python3 clean_and_token_text.py /media/hd1/clean_text/math* joined_math --phrases_file /media/hd1/glossary/v3/math*/*.xml.gz  --num_phrases 2500

    '''

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('in_files', type=str, nargs='+',
            help='One or more path to text file that needs normalization and phrase joining')
    parser.add_argument('out_dir', type=str,
            help='path to file to write the normalized text if empty prints to shell')
    parser.add_argument('--phrases_file', default=None, type=str, nargs='+',
            help='XML file with the phrases to be joined')
    parser.add_argument('--norm_args', nargs='?', default=['rm_punct'],
            help='arguments for the tokenization function')
    parser.add_argument('--num_phrases', type=int, default=None,
            help='Max number of phrases to use')
    #parser.add_argument('--skip_n', default=1, type=int)
    args = parser.parse_args()


    print('Started reading the phrases...')
    phrases_cnt = Counter()
    if args.phrases_file is not None:
        for xml_path in tqdm(args.phrases_file):
            root = etree.parse(xml_path)
            phrases_list_temp = [normalize_text(r.text)\
                    for r in root.findall('//dfndum') ]
            phrases_cnt.update([r for r in phrases_list_temp if len(r.split()) > 1])
        print('Joining {} phrases found'.format(len(phrases_cnt)))
    else:
        print('No phrases selected :(')

    phrases_list = [ph[0] for ph in phrases_cnt.most_common()]
    for ph in phrase_blacklist:
        try:
            phrases_list.remove(ph)
        except ValueError:
            print(f"phrase {ph} not in the phrase list")
    phrases_list = phrases_list[:args.num_phrases]
    os.makedirs(args.out_dir, exist_ok=True)

    with mp.Pool(processes=5, maxtasksperchild=1) as pool:
        arg_lst = []
        for infile in args.in_files:
            fname = os.path.basename(infile).split('.')[0] 
            out_file = os.path.join(args.out_dir, fname)
            arg_lst.append((infile, out_file, phrases_list)) 
        pool.starmap(tokenize_and_write, arg_lst)


