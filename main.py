import argparse
import torch

def get_args(args=None):
    parser = argparse.ArgumentParser(description="CREMAD Speech Project")

    # Dataset and paths
    parser.add_argument('--dataset', type=str, default='CREMAD')
    parser.add_argument('--modulation', type=str, default='OGM_GE')
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['sum', 'concat', 'film', 'gated'])
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--audio_path', type=str, default='/kaggle/input/cremad-1/cremad/AudioWAV/')
    parser.add_argument('--visual_path', type=str, default='/kaggle/input/cremad-1/cremad/')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--lr_decay_step', type=int, default=70)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--ckpt_path', type=str, default='/kaggle/working/ckpt')
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--use_tensorboard', type=bool, default=False)
    parser.add_argument('--tensorboard_path', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # Input dimensions

    
    # Number of training epochs
    parser.add_argument('--epoch', type=int, default=100)
    
    # Model weights and save paths
    parser.add_argument('--weight_visual', type=str, default='/kaggle/input/weight-cremad/weight.pth')
    parser.add_argument('--weight_audio', type=str, default='/kaggle/input/weight-asr/audioset_10_10_0.4593.pth')
    parser.add_argument('--save_path', type=str, default='/kaggle/working/weights/model.pth')
    
    # Train/test CSV files
    parser.add_argument('--train_csv', type=str, default='/kaggle/working/Speech_project_Vin/data/CREMAD/train.csv')
    parser.add_argument('--test_csv', type=str, default='/kaggle/working/Speech_project_Vin/data/CREMAD/test.csv')
    
    # PatchEmbed parameters
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--in_chans', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=768)
    
    # ASTModel parameters
    parser.add_argument('--fstride', type=int, default=10)
    parser.add_argument('--tstride', type=int, default=10)
    parser.add_argument('--input_tdim', type=int, default=256)
    parser.add_argument('--input_fdim', type=int, default=128)
    parser.add_argument('--imagenet_pretrain', type=bool, default=False)
    parser.add_argument('--audioset_pretrain', type=bool, default=True)
    parser.add_argument('--model_size', type=str, default='base384')
    
    # Visual model parameters
    parser.add_argument('--reduction_ratio', type=int, default=16)
    parser.add_argument('--pool_types', nargs='+', default=['avg', 'max'])
    
    # MANet parameters
    parser.add_argument('--manet_layers', nargs='+', type=int, default=[2, 2, 2, 2])
    parser.add_argument('--manet_num_classes', type=int, default=12666)
    
    # Fusion method parameters
    parser.add_argument('--fusion_input_dim', type=int, default=1088)
    parser.add_argument('--fusion_output_dim', type=int, default=7)
    parser.add_argument('--film_x_film', type=bool, default=False)
    parser.add_argument('--gated_x_gate', type=bool, default=False)
    
    # Training parameters
    parser.add_argument('--training_learning_rate', type=float, default=0.0002)
    parser.add_argument('--training_lr_decay_step', type=int, default=70)
    parser.add_argument('--training_lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--training_epoch', type=int, default=100)
    parser.add_argument('--training_save_path', type=str, default='/kaggle/working/model.pth')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    
    return args


    return args

  
if __name__ == "__main__":
    args = get_args()
    print(args.device)
    print(args.fusion_type)
    print('done')
