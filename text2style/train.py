from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
import os
import sys
# import tensorboardX
import torch.utils.tensorboard as tensorboard
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/text2style_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT")
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

#TODO sample的输入是image的batch_list, 这和我定义的sample不一样，需要仿照原trainer.py修改我的sample函数
# train_display_images_a = torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda()
# train_display_images_b = torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda()
# test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda()
# test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboard.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
while True:
    for it, (images, texts) in enumerate(zip(train_loader_a, train_loader_b)):
        trainer.update_learning_rate()
        images, texts = images.cuda().detach(), texts#我不需要计算images和texts的梯度
        # breakpoint()
        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.gen_update(images, texts, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

