import argparse

args = argparse.Namespace(
    dataset='CREMAD',
    modulation='OGM_GE',
    fusion_method='concat',
    fps=1,
    num_classes = 7,
    audio_path='./cremad-1/cremad/AudioWAV/', # fix link dataset 
    visual_path='./cremad-1/cremad/', # fix link dataset 
    batch_size=16,
    epochs=30,
    optimizer='sgd',  # Changed to 'sgd'
    learning_rate=0.0002,
    lr_decay_step=70,
    lr_decay_ratio=0.1,
    momentum=0.9,  # Added momentum argument
    ckpt_path='./ckpt',
    train=True,
    use_tensorboard=False,
    tensorboard_path=None,
    random_seed=0,
    input_tdim = 256,
    epoch = 100,
    weight_visual = './weight-cremad/weight.pth',
    weight_audio = './weight-asr/audioset_10_10_0.4593.pth',
    save_path = './weight.pth' # fix link save weight
)