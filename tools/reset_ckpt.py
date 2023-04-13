import torch
import argparse
def save_ckpt(ckpt, save_name):
    torch.save(ckpt, save_name)
    print('save changed ckpt to {}'.format(save_name))


def reset_ckpt(ckpt):
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0

def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src', type=str, default='',
                        help='Path to the main checkpoint')
    parser.add_argument('--save-name', type=str, default='',
                        help='Save directory')    
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    #print(args)
    ckpt = torch.load(args.src)
    reset_ckpt(ckpt)
    save_ckpt(ckpt, args.save_name)

if __name__ == '__main__':
    main()