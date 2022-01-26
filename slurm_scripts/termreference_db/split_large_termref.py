import pickle
from os.path import join

def chunk(Parts, List_l, zf=2):
    assert Parts <= List_l
    assert Parts > 0
    s = List_l//(Parts)
    slice_lst = [slice(s*k, s*(k+1)) for k in range(Parts-1)]
    slice_lst.append(slice(s*(Parts-1), s*(Parts) + List_l%(Parts)))
    names_lst = [repr(r+1).zfill(zf) for r in range(Parts)]
    return zip(slice_lst, names_lst)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', type=str, 
            help='This is the source of the data ex. $NEOC/termreference_db17-33_22-01')
    parser.add_argument('output_dir', type=str, 
            help='previously created dir to save the list slices')
    parser.add_argument('--termref', type=str,
            help='Term reference file pickle')
    parser.add_argument('-P', '--parts', type=int, default=100,
            help="Split the list into this many parts.")
    parser.add_argument('--prefix', type=str, default='termref',
            help="Start the split files with this string ex. termref_03.pickle")
    args = parser.parse_args()

    print('Loading the list')
    with open(join(args.source_dir, 'term_ref_lst.pickle'), 'rb') as fobj:
        trl = pickle.load(fobj)
    print('The list has been loaded!')

    for sl, name in chunk(args.parts, len(trl)):
        with open(join(args.source_dir, args.output_dir, args.prefix+'_'+name+'.pickle'), 'wb') as fobj:
            pickle.dump(trl[sl], fobj)
            
if __name__ == '__main__':
    '''
    python3 split_large_termref.py ~/rm_me/output/ --termref culito -P 4 --prefix culito
    '''
    main()
