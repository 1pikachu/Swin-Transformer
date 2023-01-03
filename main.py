# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
# import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + \
            '-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, required=False, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    # OOB
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--device', default='cpu', type=str, help='cpu, cuda or xpu')
    parser.add_argument('--nv_fuser', action='store_true', default=False, help='enable nv fuser')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    # model.cuda()
    model = model.to(args.device)
    if args.channels_last and args.device != "xpu":
        model = model.to(memory_format=torch.channels_last)
        print("---- use NHWC format")
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
    if args.device == "xpu":
        model, optimizer = torch.xpu.optimize(model=model, optimizer=optimizer, dtype=datatype)

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    criterion = criterion.to(args.device)
    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    #for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
    for epoch in range(1):
        #data_loader_train.sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad()

        if args.profile and args.device == "xpu":
            train_one_epoch_profileXPU(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler, args, datatype)
        elif args.profile:
            train_one_epoch_profile(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler, args, datatype)
        else:
            train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler, args, datatype)
        #if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
        #    save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
        #                    logger)

        #acc1, acc5, loss = validate(config, data_loader_val, model)
        #logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        #max_accuracy = max(max_accuracy, acc1)
        #logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch_profileXPU(config, model, criterion, data_loader,
        optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, args, datatype):
    num_steps = len(data_loader)
    total_time = 0.0
    total_count = 0

    for idx, (samples, targets) in enumerate(data_loader):
        if idx >= args.num_iter:
            break
        start_time = time.time()
        samples = samples.to(args.device)
        targets = targets.to(args.device)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=True, record_shapes=False) as prof:
            with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        loss = loss.cpu()
        outputs = outputs.cpu()
        torch.xpu.synchronize()
        duration = time.time() - start_time
        print("iteration:{}, training time: {} sec.".format(idx, duration))
        if idx >= args.num_warmup:
            total_time += duration
            total_count += 1
        if args.profile and idx == profile_len:
            import pathlib
            timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
            if not os.path.exists(timeline_dir):
                try:
                    os.makedirs(timeline_dir)
                except:
                    pass
            torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                timeline_dir+'profile.pt')
            torch.save(prof.key_averages(group_by_input_shape=True).table(),
                timeline_dir+'profile_detail.pt')
            torch.save(prof.table(sort_by="id", row_limit=100000),
                timeline_dir+'profile_detail_withId.pt')
            prof.export_chrome_trace(timeline_dir+"trace.json")
    batch_size = args.batch_size
    avg_time = total_time / total_count
    latency = avg_time / batch_size * 1000
    perf = batch_size / avg_time
    print("total time:{}, total count:{}".format(total_time, total_count))
    print('%d epoch training latency: %6.2f ms'%(0, latency))
    print('%d epoch training Throughput: %6.2f fps'%(0, perf))

def train_one_epoch_profile(config, model, criterion, data_loader,
        optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, args, datatype):
    num_steps = len(data_loader)
    total_time = 0.0
    total_count = 0

    profile_len = min(len(data_loader), args.num_iter) // 2
    if args.device == "cuda":
        profile_act = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    else:
        profile_act = [torch.profiler.ProfilerActivity.CPU]

    with torch.profiler.profile(
        activities=profile_act,
        record_shapes=True,
        schedule=torch.profiler.schedule(
            wait=profile_len,
            warmup=2,
            active=1,
        ),
        on_trace_ready=trace_handler,
    ) as p:
        for idx, (samples, targets) in enumerate(data_loader):
            if idx >= args.num_iter:
                break
            if args.channels_last:
                samples = samples.to(memory_format=torch.channels_last)

            start_time = time.time()
            samples = samples.to(args.device)
            targets = targets.to(args.device)

            if mixup_fn is not None:
                samples, targets = mixup_fn(samples, targets)
            if args.device == "cuda":
                with torch.cuda.amp.autocast(enabled=True, dtype=datatype):
                    outputs = model(samples)
            else:
                with torch.cpu.amp.autocast(enabled=True, dtype=datatype):
                    outputs = model(samples)
            loss = criterion(outputs, targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()
                lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
            loss_scale_value = loss_scaler.state_dict()["scale"]

            loss = loss.cpu()
            outputs = outputs.cpu()
            if args.device == "cuda":
                torch.cuda.synchronize()
            duration = time.time() - start_time
            print("iteration:{}, training time: {} sec.".format(idx, duration))
            p.step()
            if idx >= args.num_warmup:
                total_time += duration
                total_count += 1
    batch_size = args.batch_size
    avg_time = total_time / total_count
    latency = avg_time / batch_size * 1000
    perf = batch_size / avg_time
    print("total time:{}, total count:{}".format(total_time, total_count))
    print('%d epoch training latency: %6.2f ms'%(0, latency))
    print('%d epoch training Throughput: %6.2f fps'%(0, perf))

def train_one_epoch(config, model, criterion, data_loader,
        optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, args, datatype):
    num_steps = len(data_loader)
    total_time = 0.0
    total_count = 0

    for idx, (samples, targets) in enumerate(data_loader):
        if idx >= args.num_iter:
            break
        if args.channels_last and args.device != "xpu":
            samples = samples.to(memory_format=torch.channels_last)
        start_time = time.time()
        samples = samples.to(args.device)
        targets = targets.to(args.device)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if args.device == "cuda":
            with torch.cuda.amp.autocast(enabled=True, dtype=datatype):
                outputs = model(samples)
        elif args.device == "xpu":
            with torch.xpu.amp.autocast(enabled=True, dtype=datatype):
                outputs = model(samples)
        else:
            with torch.cpu.amp.autocast(enabled=True, dtype=datatype):
                outputs = model(samples)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        loss = loss.cpu()
        outputs = outputs.cpu()
        if args.device == "cuda":
            torch.cuda.synchronize()
        elif args.device == "xpu":
            torch.xpu.synchronize()
        duration = time.time() - start_time
        print("iteration:{}, training time: {} sec.".format(idx, duration))
        if idx >= args.num_warmup:
            total_time += duration
            total_count += 1
    batch_size = args.batch_size
    avg_time = total_time / total_count
    latency = avg_time / batch_size * 1000
    perf = batch_size / avg_time
    print("total time:{}, total count:{}".format(total_time, total_count))
    print('%d epoch training latency: %6.2f ms'%(0, latency))
    print('%d epoch training Throughput: %6.2f fps'%(0, perf))

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    #torch.cuda.manual_seed(seed)
    #cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, args)
