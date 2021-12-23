import numpy as np
from lxml import etree
import gzip

import sys,inspect,os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from classifier_trainer.trainer import stream_arxiv_paragraphs
import parsing_xml as px
from extract import Definiendum
import peep_tar as peep
import classifier_models as M


class fake_clf:
    def predict(arg, *args, **kwargs):
        #import pdb; pdb.set_trace()
        return np.array([1.0 for l in arg])

class Vectorizer():
    def __init__(self):
        pass
    def transform(self, L):
        return [1 for i in L]

def untar_clf_append(tfile, out_path, clf, vzer, thresh=0.5, min_words=15):
    '''
    Arguments:
    `tfile` tarfile with arxiv format ex. 1401_001.tar.gz
    `out_path` directory to save the xml.tar.gz file with the same name as `tfile`
    `clf` model with .predict() attribute
    `vzer` funtion that take the text of a paragraph and outputs padded np.arrays for `clf`
    '''
    opt_prob = 0.1
    for fname, tar_fobj in peep.tar_iter(tfile, '.xml'):
        root = etree.Element('root')
        #print(f"**Peeping into file {fname}  **")
        try:
            DD = px.DefinitionsXML(tar_fobj) 
            if DD.det_language() in ['en', None]:
                art_tree = Definiendum(DD, clf, None, vzer,\
                        None, fname=fname, thresh=opt_prob,\
                        min_words=15).root
                if art_tree is not None: root.append(art_tree)

            just_the_name = os.path.splitext(fname)[0]
            just_the_name = os.path.basename(just_the_name)
            with gzip.open(os.path.join(out_path,just_the_name+'.gz'), 'wb') as out_f:
                #print("Writing to dfdum zipped file to: %s"%gz_out_path)
                #raise etree.SerialisationError('-- ERROR --')
                out_f.write(etree.tostring(root, pretty_print=True))
        except ValueError as ee:
            print(f"{repr(ee)}, 'file: ', {fname}, ' is empty'")
        except etree.SerialisationError as ee:
            print(f"{repr(ee)}, 'file: ', {fname}, ' IS NOT WRITABLE.'")
    return root


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mine', type=str,
            help='Path to save the logs and the results of mining. ex. math03 math04')
    parser.add_argument('out_path', type=str,
            help='Path to save the logs and the results of mining. ex. math03 math04')
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    untar_clf_append(args.mine, args.out_path, fake_clf, Vectorizer())

