import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mmimdb', type=str, help='Datasets.')
parser.add_argument('--root', default='./datasets', type=str, help='Root of datasets')
parser.add_argument('--target', default='./datasets', type=str, help='Root of datasets')
args = parser.parse_args()

if args.dataset.lower() == 'mmimdb':
    from clip.utils.write_mmimdb import make_arrow
    make_arrow(args.root, args.target)
    
elif args.dataset.lower() == 'food101':
    from clip.utils.write_food101 import make_arrow
    make_arrow(args.root, args.target)
    
elif args.dataset.lower() == 'hateful_memes':
    from clip.utils.write_hatememes import make_arrow
    make_arrow(args.root, args.target)