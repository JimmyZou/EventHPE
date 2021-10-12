import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import time
from tensorboardX import SummaryWriter
import sys
sys.path.append('../')
from flow_net.model import OpticalFlowNet
from flow_net.dataloader import FlowDataloader, Augmentation
from flow_net.loss_funcs import compute_photometric_loss, compute_smoothness_loss
from flow_net.flowlib import flow_to_image, flow_viz_np
import collections
import numpy as np


def train(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set dataset
    dataset_train = FlowDataloader(data_dir=args.data_dir, input_channel=args.input_channel, num_skip=args.num_skip,
                                   skip=args.skip, max_num_events=args.max_num_events, transform=Augmentation(),
                                   mode='train', img_size=args.img_size, source=args.source)
    train_generator = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=False)
    total_iters = len(dataset_train) // args.batch_size + 1

    dataset_val = FlowDataloader(data_dir=args.data_dir, input_channel=args.input_channel,
                                 num_skip=1, skip=args.skip, max_num_events=args.max_num_events, transform=None,
                                 mode='test', img_size=args.img_size, source=args.source)
    val_generator = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=False)

    # set model
    model = OpticalFlowNet(input_channel=args.input_channel, output_channel=2, num_layers=4, base_channel=32)

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # set tensorboard
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter('%s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    print('[tensorboard] %s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    model_dir = '%s/%s/%s/model_flow.pkl' % (args.result_dir, args.log_dir, start_time)

    # training
    best_loss = 1000
    for epoch in range(args.epochs + 1):
        print('====================================== Epoch %i ========================================' % (epoch + 1))

        # train
        print('------------------------------------- Training ------------------------------------')
        model.train()
        results = collections.defaultdict(list)
        start_time = time.time()
        for iter, batch_data in enumerate(train_generator):
            # batch_data: {fullpic1, fullpic2, events_frame, dt}
            img1 = batch_data['img1'].to(device=device, dtype=dtype)
            img2 = batch_data['img2'].to(device=device, dtype=dtype)
            events_frame = batch_data['events_frame'].to(device=device, dtype=dtype)
            dt = batch_data['dt'].to(device=device, dtype=dtype).view(img1.size()[0], 1, 1, 1)

            optimizer.zero_grad()
            if args.source == 'events':
                multi_scale_flows = model(events_frame, dt)
            elif args.source == 'full_pic':
                multi_scale_flows = model(torch.cat([img1, img2], dim=1) / 255., dt)
            else:
                multi_scale_flows = model(torch.cat([img1, img2], dim=1) / 255., dt)

            photometric_loss = compute_photometric_loss(multi_scale_flows, img1, img2)
            smoothness_loss = compute_smoothness_loss(multi_scale_flows)
            loss = photometric_loss + args.smoothness_loss_weight * smoothness_loss

            loss.backward()
            optimizer.step()

            # collect results
            results['scalar/photometric_loss'].append(photometric_loss.detach())
            results['scalar/smoothness_loss'].append(smoothness_loss.detach())
            results['scalar/loss'].append(loss.detach())

            # if iter > 4:
            #     break
            if iter % (total_iters // 10) == 0:
            # if iter % 2 == 0:
                results['flow'] = multi_scale_flows[-1].detach()
                results['img1'] = img1.detach()
                results['img2'] = img2.detach()
                results['events'] = events_frame

                results['photometric_loss'] = torch.mean(torch.stack(results['scalar/photometric_loss'], dim=0))
                results['smoothness_loss'] = torch.mean(torch.stack(results['scalar/smoothness_loss'], dim=0))
                results['loss'] = torch.mean(torch.stack(results['scalar/loss'], dim=0))
                progress = 10 * iter // (total_iters // 10) + 1
                write_tensorboard(writer, results, epoch, progress, 'train')

                end_time = time.time()
                time_used = (end_time - start_time) / 60.
                print('>>> [epoch {:2d}/ iter {:6d}] {:3d}%\n'
                      '    loss: {:.4f}, photometric_loss: {:.4f}, smoothness_loss: {:.4f}\n'
                      '    lr: {:.6f}, time used: {:.2f} mins, still need time for this epoch: {:.2f} mins.'
                      .format(epoch, iter, progress - 1, results['loss'],
                              results['photometric_loss'], results['smoothness_loss'],
                              scheduler.get_last_lr()[0], time_used, (100 / progress - 1) * time_used))

        # '''
        # test
        print('------------------------------------- test ------------------------------------')
        start_time = time.time()
        model.eval()  # dropout layers will not work in eval mode
        results = collections.defaultdict(list)
        with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
            for idx, batch_data in enumerate(val_generator):
                # batch_data: {img1, img2, events_frame, dt}
                img1 = batch_data['img1'].to(device=device, dtype=dtype)
                img2 = batch_data['img2'].to(device=device, dtype=dtype)
                events_frame = batch_data['events_frame'].to(device=device, dtype=dtype)
                dt = batch_data['dt'].to(device=device, dtype=dtype).view(img1.size()[0], 1, 1, 1)

                if args.source == 'events':
                    multi_scale_flows = model(events_frame, dt)
                elif args.source == 'full_pic':
                    multi_scale_flows = model(torch.cat([img1, img2], dim=1) / 255., dt)
                else:
                    multi_scale_flows = model(torch.cat([img1, img2], dim=1) / 255., dt)
                photometric_loss = compute_photometric_loss(multi_scale_flows, img1, img2)
                smoothness_loss = compute_smoothness_loss(multi_scale_flows)
                loss = photometric_loss + args.smoothness_loss_weight * smoothness_loss

                # collect results
                results['scalar/photometric_loss'].append(photometric_loss.detach())
                results['scalar/smoothness_loss'].append(smoothness_loss.detach())
                results['scalar/loss'].append(loss.detach())

                if idx % 400 == 0:
                    results['flow'].append(multi_scale_flows[-1].detach())
                    results['img1'].append(img1.detach())
                    results['img2'].append(img2.detach())
                    results['events'].append(events_frame)

            results['photometric_loss'] = torch.mean(torch.stack(results['scalar/photometric_loss'], dim=0))
            results['smoothness_loss'] = torch.mean(torch.stack(results['scalar/smoothness_loss'], dim=0))
            results['loss'] = torch.mean(torch.stack(results['scalar/loss'], dim=0))
            results['flow'] = torch.cat(results['flow'], dim=0)
            results['img1'] = torch.cat(results['img1'], dim=0)
            results['img2'] = torch.cat(results['img2'], dim=0)
            results['events'] = torch.cat(results['events'], dim=0)
            write_tensorboard(writer, results, epoch, 0, 'test')

            end_time = time.time()
            time_used = (end_time - start_time) / 60.
            print('>>> loss: {:.4f}, photometric_loss: {:.4f}, smoothness_loss: {:.4f}\n'
                  '    time used: {:.2f} mins'
                  .format(results['loss'], results['photometric_loss'], results['smoothness_loss'], time_used))

            if best_loss > results['loss']:
                torch.save(model.state_dict(), model_dir)
                best_loss = results['loss']
                print('>>> Model saved as {}... best loss {:.4f}'.format(model_dir, best_loss))
        # '''
        # break
        scheduler.step()
    writer.close()


def write_tensorboard(writer, results, epoch, progress, mode):
    writer.add_scalar('%s/photometric_loss' % mode, results['photometric_loss'], epoch*100+progress)
    writer.add_scalar('%s/smoothness_loss' % mode, results['smoothness_loss'], epoch*100+progress)
    writer.add_scalar('%s/loss' % mode, results['loss'], epoch*100+progress)

    img1 = results['img1'].cpu().numpy()
    img2 = results['img2'].cpu().numpy()
    flow = results['flow'].cpu().numpy()
    events = results['events'].float().cpu().numpy()

    if np.max(img1) > 1:
        img1 = img1 / 255.
        img2 = img2 / 255.

    n = img1.shape[0]
    for i in range(n):
        writer.add_images('%s/example%i/img1' % (mode, i+1), img1[i], epoch*100+progress, dataformats='CHW')
        flow_rgb = np.transpose(flow_to_image(np.transpose(flow[i], [1, 2, 0])), [2, 0, 1])
        # flow_rgb = np.transpose(flow_viz_np(flow[i]), [2, 0, 1])
        if len(events.shape) > 3:
            event_frame = (np.sum(events[i], axis=0, keepdims=True) > 0).astype(np.float32)
            writer.add_images('%s/example%i/events' % (mode, i+1), event_frame, epoch*100+progress, dataformats='CHW')
        writer.add_images('%s/example%i/flow' % (mode, i+1), flow_rgb, epoch*100+progress, dataformats='CHW')
        writer.add_images('%s/example%i/img2' % (mode, i+1), img2[i], epoch*100+progress, dataformats='CHW')


def get_args():
    def print_args(args):
        """ Prints the argparse argmuments applied
        Args:
          args = parser.parse_args()
        """
        _args = vars(args)
        max_length = max([len(k) for k, _ in _args.items()])
        for k, v in _args.items():
            print(' ' * (max_length - len(k)) + k + ': ' + str(v))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='3')
    parser.add_argument('--data_dir', type=str, default='/home/shihao/data_event')
    parser.add_argument('--result_dir', type=str, default='/home/shihao/exp_flow')
    parser.add_argument('--log_dir', type=str, default='log')

    parser.add_argument('--input_channel', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--smoothness_loss_weight', type=float, default=0.5)
    parser.add_argument('--num_skip', type=int, default=4)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--max_num_events', type=int, default=60000)
    parser.add_argument('--source', type=str, default='events')

    parser.add_argument('--batch_size', '-bs', type=int, default=16)
    parser.add_argument('--epochs', '-epochs', type=int, default=30)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=float, default=3)
    args = parser.parse_args()
    print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    train(args)


if __name__ == '__main__':
    main()

