import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='train_audio')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--audio_path', type=str, default='CREMA-D/AudioWAV/', help='Path to audio files')
    parser.add_argument('--visual_path', type=str, default='CREMA-D/VideoFlashFPS30/', help='Path to visual files')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--save_path', type=str, default='/content/kaggle-working/', help='Path to save the model')
    args = parser.parse_args()
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(args.device)
