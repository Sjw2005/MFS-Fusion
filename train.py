import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from data.test_dataloder import Dataset as test_dataset
from models.select_model import define_Model
import warnings
from models.loss_vif import fusion_loss_vif
warnings.filterwarnings("ignore")
testloss = fusion_loss_vif()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
best_loss =100000000.0
def main(json_path='config.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        for key, path in opt['path'].items():
            print(path)
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    border = opt['scale']

    if opt['rank'] == 0:
        option.save(opt)
    opt = option.dict_to_nonedict(opt)
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = None
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True,
                                                   seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'] // opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers'] // opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            # test_set = test_dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    global best_loss
    model = define_Model(opt)
    model.init_train()

    # train loop
    if opt['datasets']['dataset_type'] in ['mef_GT', 'mff_GT']:
        need_GT = True
    else:
        need_GT = False
    for epoch in range(300):  # keep running
        for i, train_data in enumerate(train_loader):
            # train_data['A']/train_data['B'] from DataLoader:
            # [B, 1, H_size, H_size], default [2, 1, 256, 256]

            model.update_learning_rate(epoch)
            model.feed_data(train_data, need_GT=need_GT)
            model.optimize_parameters(epoch)
        logs = model.current_log()  # such as loss
        message = '<epoch:{:3d}, lr:{:.3e}> '.format(epoch, model.current_learning_rate())
        for k, v in logs.items():  # merge log information into message
                message += '{:s}: {:.3e} '.format(k, v)
        logger.info(message)

        if epoch % 1 == 0:

            avg_psnr = 0.0
            idx = 0
            img_dir = os.path.join(opt['path']['images'], '{:d}'.format(epoch))
            loss = 0.0
            for test_data in test_loader:
                idx += 1
                image_name_ext = os.path.basename(test_data['A_path'][0])
                img_name, ext = os.path.splitext(image_name_ext)

                    # img_dir = os.path.join(opt['path']['images'], img_name)
                util.mkdir(img_dir)

                model.feed_data(test_data, phase='test')
                A_image,B_image,fusion_image = model.test()
                # Test tensors are batched with batch_size=1:
                # A_image/B_image/fusion_image: [1, 1, H, W]
                fusion_loss, loss_gradient, loss_l1 = testloss(A_image,B_image, fusion_image)
                loss += fusion_loss
                visuals = model.current_visuals(need_H=need_GT)
                    # E_img = util.tensor2uint(visuals['E'])
                img=visuals['E']
                # visuals['E'] is one sample without batch dim:
                # img: [1, H, W]
                E_img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
                # squeeze() removes channel dim for grayscale output:
                # E_img: [H, W] (if grayscale) or [C, H, W] (if multi-channel)
                if E_img.ndim == 3:
                    E_img = np.transpose(E_img, (1, 2, 0))
                    # CHW -> HWC when tensor is multi-channel
                E_img=np.uint8((E_img*255.0).round())
                # Final saved array:
                # E_img: [H, W] uint8 (grayscale) or [H, W, C] uint8
                save_img_path = os.path.join(img_dir, '{:s}.png'.format(img_name))
                util.imsave(E_img, save_img_path)
            a_loss=loss / idx
            if a_loss<best_loss:
                best_loss = a_loss
                print("epoch:{} Best loss {} :".format(epoch, best_loss))
                save_dir = opt['path']['models']
                save_filename = '{}_{}.pth'.format(epoch, 'GG')
                save_path = os.path.join(save_dir, save_filename)
                # logger.info('Saving the model. Save path is:{}'.format(save_path))
                model.save(epoch)



if __name__ == '__main__':
    main()
