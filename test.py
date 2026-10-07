import argparse
import os

import cv2
import numpy as np
import torch.nn.functional as FF
import torch.nn.parallel
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import utils
from base.parse_config import ConfigParser
from dataset.dataloader import InpaintingDataset
from dnnlib.util import get_obj_by_name
from inpainting_metric import get_inpainting_metrics
from trainers.nms_temp import get_nms as get_np_nms
from trainers.pl_trainers import wf_inference_test


def main(args, config):
    # build models architecture, then print to console
    structure_upsample = get_obj_by_name(config['structure_upsample_class'])()

    edgeline_tsr = get_obj_by_name(config['edgeline_tsr_class'])()
    grad_tsr = get_obj_by_name(config['grad_tsr_class'])()
    ftr = get_obj_by_name(config['g_class'])(config=config['g_args'])
    D = get_obj_by_name(config['d_class'])(config=config['d_args'])

    if 'PLTrainer' not in config.config or config['PLTrainer'] is None:
        config.config['PLTrainer'] = 'trainers.pl_trainers.FinetunePLTrainer'

    model = get_obj_by_name(config['PLTrainer'])(structure_upsample, edgeline_tsr, grad_tsr, ftr, D, config,
                                                 'ckpts/' + args.exp_name, use_ema=args.use_ema, dynamic_size=args.dynamic_size, test_only=True)

    if args.use_ema:
        model.reset_ema()

    if args.ckpt_resume:
        print("Loading checkpoint: {} ...".format(args.ckpt_resume))
        checkpoint = torch.load(args.ckpt_resume, map_location='cpu')
        utils.torch_init_model(model, checkpoint, key='state_dict')

    def adapter_state_from_checkpoint(path, source, prefix):
        adapter_checkpoint = torch.load(path, map_location='cpu')
        if adapter_checkpoint.get('format') == 'zitspp_joint_peft_compact_v1':
            source_state = adapter_checkpoint[source]
            return {
                key[len(prefix):]: value
                for key, value in source_state.items()
                if key.startswith(prefix)
            }
        state_dict = adapter_checkpoint.get('state_dict', adapter_checkpoint)
        full_prefix = '{}.{}'.format(source, prefix)
        return {
            key[len(full_prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(full_prefix)
        }

    if args.sa_adapter_ckpt:
        adapter_state = adapter_state_from_checkpoint(
            args.sa_adapter_ckpt,
            args.sa_adapter_source,
            'GCs.spatial_adapters.',
        )
        active_ftr = model.ftr_ema if args.use_ema else model.ftr
        expected = set(active_ftr.GCs.spatial_adapters.state_dict())
        if set(adapter_state) != expected:
            raise RuntimeError(
                'SA adapter checkpoint mismatch: missing={} unexpected={}'.format(
                    sorted(expected - set(adapter_state)),
                    sorted(set(adapter_state) - expected),
                )
            )
        current = active_ftr.GCs.spatial_adapters.state_dict()
        with torch.no_grad():
            for key, value in adapter_state.items():
                current[key].copy_(value)
        print("Loaded SA adapter ({}): {}".format(args.sa_adapter_source, args.sa_adapter_ckpt))

    if args.dense_decoder_adapter_ckpt:
        adapter_state = adapter_state_from_checkpoint(
            args.dense_decoder_adapter_ckpt,
            args.dense_decoder_adapter_source,
            'G.dense_guidance_adapters.',
        )
        active_ftr = model.ftr_ema if args.use_ema else model.ftr
        expected = set(active_ftr.G.dense_guidance_adapters.state_dict())
        if set(adapter_state) != expected:
            raise RuntimeError(
                'Dense decoder adapter checkpoint mismatch: missing={} unexpected={}'.format(
                    sorted(expected - set(adapter_state)),
                    sorted(set(adapter_state) - expected),
                )
            )
        current = active_ftr.G.dense_guidance_adapters.state_dict()
        with torch.no_grad():
            for key, value in adapter_state.items():
                current[key].copy_(value)
        print(
            "Loaded dense decoder adapter ({}): {}".format(
                args.dense_decoder_adapter_source,
                args.dense_decoder_adapter_ckpt,
            )
        )

    if args.tsr_adapter_ckpt:
        adapter_checkpoint = torch.load(args.tsr_adapter_ckpt, map_location='cpu')
        adapter_state = adapter_checkpoint[args.tsr_adapter_source]
        expected = set(model.edgeline_tsr.mask_adapters.state_dict())
        if set(adapter_state) != expected:
            raise RuntimeError(
                'TSR adapter checkpoint mismatch: missing={} unexpected={}'.format(
                    sorted(expected - set(adapter_state)),
                    sorted(set(adapter_state) - expected),
                )
            )
        current = model.edgeline_tsr.mask_adapters.state_dict()
        with torch.no_grad():
            for key, value in adapter_state.items():
                current[key].copy_(value)
        print("Loaded TSR mask adapter ({}): {}".format(args.tsr_adapter_source, args.tsr_adapter_ckpt))

    if hasattr(model, "wf"):
        model.wf.load_state_dict(torch.load(args.wf_ckpt, map_location='cpu')['model'])

    model.cuda()

    if args.use_ema:
        model.ftr_ema.eval()
    else:
        model.ftr.eval()

    os.makedirs(args.save_path, exist_ok=True)

    # dataset
    dataset = InpaintingDataset(args.img_dir, args.mask_dir, test_size=args.test_size, use_gradient=config['g_args']['use_gradient'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch['size_ratio'] = -1
            batch['H'] = -1
            for k in batch:
                if type(batch[k]) is torch.Tensor:
                    batch[k] = batch[k].cuda()

            # load line
            batch['line_256'] = wf_inference_test(model.wf, batch['img_512'], h=256, w=256, masks=batch['mask_512'],
                                                  valid_th=0.85, mask_th=0.85, obj_remove=args.obj_removal)
            imgh = batch['imgh'][0].item()
            imgw = batch['imgw'][0].item()

            # inapint prior
            edge_pred, line_pred = model.edgeline_tsr.forward(batch['img_256'], batch['line_256'], masks=batch['mask_256'])
            line_pred = batch['line_256'] * (1 - batch['mask_256']) + line_pred * batch['mask_256']

            edge_pred = edge_pred.detach()
            line_pred = line_pred.detach()

            current_size = 256
            if current_size != min(imgh, imgw):
                while current_size * 2 <= max(imgh, imgw):
                    # nms for HR
                    line_pred = model.structure_upsample(line_pred)[0]
                    edge_pred_nms = get_np_nms(edge_pred, binary_threshold=args.binary_threshold)
                    edge_pred_nms = model.structure_upsample(edge_pred_nms)[0]
                    edge_pred_nms = torch.sigmoid((edge_pred_nms + 2) * 2)
                    line_pred = torch.sigmoid((line_pred + 2) * 2)
                    current_size *= 2

                edge_pred_nms = FF.interpolate(edge_pred_nms, size=(imgh, imgw), mode='bilinear', align_corners=False)
                edge_pred = FF.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
                edge_pred[edge_pred >= 0.25] = edge_pred_nms[edge_pred >= 0.25]
                line_pred = FF.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
            else:
                edge_pred = FF.interpolate(edge_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)
                line_pred = FF.interpolate(line_pred, size=(imgh, imgw), mode='bilinear', align_corners=False)

                if config['g_args']['use_gradient'] is True:
                    gradientx, gradienty = model.grad_tsr.forward(batch['img_256'], batch['gradientx'], batch['gradienty'], masks=batch['mask_256'])
                    gradientx = batch['gradientx'] * (1 - batch['mask_256']) + gradientx * batch['mask_256']
                    gradienty = batch['gradienty'] * (1 - batch['mask_256']) + gradienty * batch['mask_256']
                    gradientx = FF.interpolate(gradientx, size=(imgh, imgw), mode='bilinear')
                    gradientx = gradientx * batch['mask'] + batch['gradientx_hr'] * (1 - batch['mask'])

                    gradienty = FF.interpolate(gradienty, size=(imgh, imgw), mode='bilinear')
                    gradienty = gradienty * batch['mask'] + batch['gradienty_hr'] * (1 - batch['mask'])

                    batch['gradientx'] = gradientx.detach()
                    batch['gradienty'] = gradienty.detach()

            batch['edge'] = edge_pred.detach()
            batch['line'] = line_pred.detach()

            if args.use_ema:
                gen_ema_img, _ = model.run_G_ema(batch)
            else:
                gen_ema_img, _ = model.run_G(batch)
            gen_ema_img = torch.clamp(gen_ema_img, -1, 1)
            gen_ema_img = (gen_ema_img + 1) / 2
            gen_ema_img = gen_ema_img * 255.0
            gen_ema_img = gen_ema_img.permute(0, 2, 3, 1).int().cpu().numpy()
            if not args.save_image_only:
                edge_res = (batch['edge'] * 255.0).permute(0, 2, 3, 1).int().cpu().numpy()
                edge_res = np.tile(edge_res, [1, 1, 1, 3])
                line_res = (batch['line'] * 255.0).permute(0, 2, 3, 1).int().cpu().numpy()
                line_res = np.tile(line_res, [1, 1, 1, 3])
                masked_img = (batch['image'] * (1 - batch['mask']) + 1) / 2 * 255
                masked_img = masked_img.permute(0, 2, 3, 1).int().cpu().numpy()
                if config['g_args']['use_gradient'] is True:
                    gradientx = (gradientx - gradientx.min()) / (gradientx.max() - gradientx.min() + 1e-7)
                    gradientx = np.tile((gradientx * 255.0).permute(0, 2, 3, 1).int().cpu().numpy(), [1, 1, 1, 3])
                    gradienty = (gradienty - gradienty.min()) / (gradienty.max() - gradienty.min() + 1e-7)
                    gradienty = np.tile((gradienty * 255.0).permute(0, 2, 3, 1).int().cpu().numpy(), [1, 1, 1, 3])
                    final_res = np.concatenate([masked_img, edge_res, line_res, gradientx, gradienty, gen_ema_img], axis=2)
                else:
                    final_res = np.concatenate([masked_img, edge_res, line_res, gen_ema_img], axis=2)
            else:
                final_res = gen_ema_img
            cv2.imwrite(args.save_path + '/' + batch['name'][0], final_res[0, :, :, ::-1])

    if args.eval:
        res = get_inpainting_metrics(args.eval_path, args.save_path, get_fid=True)
        for k in res:
            print(k, res[k])


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--config', default="config.yml", type=str, help='config file path')
    args.add_argument('--exp_name', default=None, type=str, help='method name')
    args.add_argument('--dynamic_size', action='store_true', help='Whether to finetune in dynamic size?')
    args.add_argument('--use_ema', action='store_true', help='Whether to use ema?')
    args.add_argument('--ckpt_resume', default='last.ckpt', type=str, help='PL path to restore')
    args.add_argument('--wf_ckpt', type=str, default=None, help='Line detector weights')
    args.add_argument('--sa_adapter_ckpt', type=str, default=None, help='External SA-ZeroRA adapter checkpoint')
    args.add_argument('--sa_adapter_source', choices=['ftr', 'ftr_ema'], default='ftr_ema')
    args.add_argument('--dense_decoder_adapter_ckpt', type=str, default=None, help='External dense decoder adapter checkpoint')
    args.add_argument('--dense_decoder_adapter_source', choices=['ftr', 'ftr_ema'], default='ftr_ema')
    args.add_argument('--dense_decoder_guidance', choices=['mask-only', 'edge', 'gradient', 'line', 'full'], default=None)
    args.add_argument('--dense_decoder_stages', nargs='+', choices=['256', '128', '64'], default=None)
    args.add_argument('--tsr_adapter_ckpt', type=str, default=None, help='External TSR mask adapter checkpoint')
    args.add_argument('--tsr_adapter_source', choices=['adapter', 'adapter_ema'], default='adapter_ema')
    args.add_argument('--save_path', default='outputs', type=str, help='path to save')
    args.add_argument('--img_dir', type=str, default=None, help='Test image path')
    args.add_argument('--mask_dir', type=str, default=None, help='Mask path')
    args.add_argument('--test_size', type=int, default=None, help='Test image size')
    args.add_argument('--binary_threshold', type=int, default=50, help='binary_threshold for E-NMS (from 0 to 255)')
    args.add_argument('--eval', action='store_true', help='Whether to eval?')
    args.add_argument('--save_image_only', action='store_true', help='Only save image')
    args.add_argument('--obj_removal', action='store_true', help='obj_removal')
    args.add_argument('--eval_path', type=str, default=None, help='Eval gt image path')

    # custom cli options to modify configuration from default values given in json file.
    args = args.parse_args()
    assert args.img_dir is not None
    assert args.mask_dir is not None
    args.resume = None
    config = ConfigParser.from_args(args, mkdir=False)
    if args.dense_decoder_guidance is not None:
        guidance_names = {
            'mask-only': (),
            'edge': ('edge',),
            'gradient': ('gradientx', 'gradienty'),
            'line': ('line',),
            'full': ('edge', 'gradientx', 'gradienty', 'line'),
        }[args.dense_decoder_guidance]
        config.config['g_args']['dense_decoder_zero_ra']['guidance_names'] = guidance_names
    if args.dense_decoder_stages is not None:
        stage_indices = tuple({'256': 0, '128': 1, '64': 2}[value] for value in args.dense_decoder_stages)
        config.config['g_args']['dense_decoder_zero_ra']['enabled_stages'] = stage_indices
    SEED = 123456
    torch.manual_seed(SEED)

    num_gpus = torch.cuda.device_count()
    args.num_gpus = num_gpus

    main(args, config)
