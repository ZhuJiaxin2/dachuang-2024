import torch
from trainer import MUNIT_Trainer
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import save_image
import PIL.Image as Image
import torch.nn.functional as F
from utils import get_config, get_all_data_loaders
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/text2style_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT")
opts = parser.parse_args()

config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

_, _, test_loader_a, test_loader_b = get_all_data_loaders(config)

model = MUNIT_Trainer(config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.resume('./outputs/text2style_folder/checkpoints', config)
model.eval()

# test_img = Image.open('./tubingen.jpg')

for it, (images, texts) in enumerate(zip(test_loader_a, test_loader_b)):
    test_img = torch.unsqueeze(images[0], 0).to(device)
    # test_text = ['The Starry Night by Vincent Van Gogh']
    test_text = ['cartoon style']
    # test_img = test_img
    output = model(test_img, test_text)

    # 将output的数据范围从[-1, 1]改为[0, 1]
    output = (output + 1) / 2
    save_image(output, f'./outputs/results_cartoon/{it}_output.png')



