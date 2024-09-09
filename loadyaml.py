import yaml

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access the parameters from the config dictionary
dataset = config['dataset']
modulation = config['modulation']
fusion_method = config['fusion_method']
fps = config['fps']
num_classes = config['num_classes']
audio_path = config['audio_path']
visual_path = config['visual_path']
batch_size = config['batch_size']
epochs = config['epochs']
optimizer = config['optimizer']
learning_rate = config['learning_rate']
lr_decay_step = config['lr_decay_step']
lr_decay_ratio = config['lr_decay_ratio']
momentum = config['momentum']
ckpt_path = config['ckpt_path']
train = config['train']
use_tensorboard = config['use_tensorboard']
tensorboard_path = config['tensorboard_path']
random_seed = config['random_seed']
input_tdim = config['input_tdim']
epoch = config['epoch']
weight_visual = config['weight_visual']
weight_audio = config['weight_audio']
save_path = config['save_path']
