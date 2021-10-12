import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import cv2
from tensorboardX import SummaryWriter
import sys
sys.path.append('../')
from event_pose_estimation.model import EventTrackNet
from event_pose_estimation.dataloader import TrackingDataloader
from event_pose_estimation.loss_funcs import compute_losses, compute_mpjpe, compute_pa_mpjpe, compute_pelvis_mpjpe
import collections
import numpy as np
import event_pose_estimation.utils as util


def train(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set dataset
    dataset_train = TrackingDataloader(
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        num_steps=args.num_steps,
        skip=args.skip,
        events_input_channel=args.events_input_channel,
        img_size=args.img_size,
        mode='train',
        use_flow=args.use_flow,
        use_flow_rgb=args.use_flow_rgb,
        use_hmr_feats=args.use_hmr_feats,
        use_vibe_init=args.use_vibe_init,
        use_hmr_init=args.use_hmr_init,
    )
    train_generator = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=args.pin_memory
    )
    total_iters = len(dataset_train) // args.batch_size + 1

    dataset_val = TrackingDataloader(
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        num_steps=args.num_steps,
        skip=args.skip,
        events_input_channel=args.events_input_channel,
        img_size=args.img_size,
        mode='test',
        use_flow=args.use_flow,
        use_flow_rgb=args.use_flow_rgb,
        use_hmr_feats=args.use_hmr_feats,
        use_vibe_init=args.use_vibe_init,
        use_hmr_init=args.use_hmr_init,
    )
    val_generator = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=args.pin_memory
    )

    if args.use_vibe_init or args.use_hmr_init:
        smpl_dir = '../smpl_model/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
    else:
        smpl_dir = '../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    print('[smpl_dir] %s' % smpl_dir)

    # set model
    model = EventTrackNet(
        events_input_channel=args.events_input_channel,
        smpl_dir=smpl_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        n_layers=args.rnn_layers,
        hidden_size=2048,
        bidirectional=False,
        add_linear=False,
        use_residual=True,
        pose_dim=24*6,
        use_flow=args.use_flow,
        vibe_regressor=args.vibe_regressor,
        cam_intr=torch.from_numpy(dataset_train.cam_intr).float()
    )
    mse_func = torch.nn.MSELoss()
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # set optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_start)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_decay_step], gamma=args.lr_decay_rate)
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # set tensorboard
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter('%s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    print('[tensorboard] %s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    if args.model_dir is not None:
        print('[model dir] model loaded from %s' % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    save_dir = '%s/%s/%s' % (args.result_dir, args.log_dir, start_time)
    model_dir = '%s/%s/%s/model_events_pose.pkl' % (args.result_dir, args.log_dir, start_time)

    # training
    best_loss = 1e4
    for epoch in range(args.epochs):
        print('====================================== Epoch %i ========================================' % (epoch + 1))
        # '''
        print('------------------------------------- Training ------------------------------------')
        model.train()
        results = collections.defaultdict(list)
        results['faces'] = model.smpl.faces
        results['cam_intr'] = model.cam_intr
        start_time = time.time()
        for iter, data in enumerate(train_generator):
            # data: {events, flows, init_shape, hidden_feats, theta, tran, joints2d, joints3d, info}
            for k in data.keys():
                if k != 'info':
                    data[k] = data[k].to(device=device, dtype=dtype)

            optimizer.zero_grad()
            if args.use_amp:
                # cast operations to mixed precision
                with torch.cuda.amp.autocast():
                    if args.use_flow:
                        if args.use_flow_rgb:
                            out = model(data['events'], data['init_shape'], data['flows_rgb'], data['hidden_feats'])
                        else:
                            out = model(data['events'], data['init_shape'], data['flows'], data['hidden_feats'])
                    else:
                        out = model(data['events'], data['init_shape'], None, data['hidden_feats'])

                    loss_dict = compute_losses(out, data, mse_func, device, args)
                    mpjpe = compute_mpjpe(out['joints3d'].detach(), data['joints3d'])  # [B, T, 24]
                    loss = loss_dict['delta_tran'] + \
                           loss_dict['tran'] + \
                           loss_dict['theta'] + \
                           loss_dict['joints3d'] + \
                           loss_dict['joints2d'] + \
                           loss_dict['flow']
                # scale the loss and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.use_flow:
                    if args.use_flow_rgb:
                        out = model(data['events'], data['init_shape'], data['flows_rgb'], data['hidden_feats'])
                    else:
                        out = model(data['events'], data['init_shape'], data['flows'], data['hidden_feats'])
                else:
                    out = model(data['events'], data['init_shape'], None, data['hidden_feats'])

                loss_dict = compute_losses(out, data, mse_func, device, args)
                mpjpe = compute_mpjpe(out['joints3d'].detach(), data['joints3d'])  # [B, T, 24]
                loss = loss_dict['delta_tran'] + \
                       loss_dict['tran'] + \
                       loss_dict['theta'] + \
                       loss_dict['joints3d'] + \
                       loss_dict['joints2d'] + \
                       loss_dict['flow']
                loss.backward()
                optimizer.step()

            # collect results
            results['scalar/delta_tran'].append(loss_dict['delta_tran'].detach())
            results['scalar/tran'].append(loss_dict['tran'].detach())
            results['scalar/theta'].append(loss_dict['theta'].detach())
            results['scalar/joints3d'].append(loss_dict['joints3d'].detach())
            results['scalar/joints2d'].append(loss_dict['joints2d'].detach())
            results['scalar/flow'].append(loss_dict['flow'].detach())
            results['scalar/loss'].append(loss.detach())
            results['scalar/mpjpe'].append(torch.mean(mpjpe.detach()))

            # if iter > 10:
            #     break
            # if iter % 2 == 0:
            display = 20
            if iter % (total_iters // display) == 0:
                results['info'] = (data['info'][0][0], data['info'][1][0])
                results['verts'] = out['verts'][0].detach()

                results['delta_tran'] = torch.mean(torch.stack(results['scalar/delta_tran'], dim=0))
                results['tran'] = torch.mean(torch.stack(results['scalar/tran'], dim=0))
                results['theta'] = torch.mean(torch.stack(results['scalar/theta'], dim=0))
                results['joints3d'] = torch.mean(torch.stack(results['scalar/joints3d'], dim=0))
                results['joints2d'] = torch.mean(torch.stack(results['scalar/joints2d'], dim=0))
                results['flow'] = torch.mean(torch.stack(results['scalar/flow'], dim=0))
                results['loss'] = torch.mean(torch.stack(results['scalar/loss'], dim=0))
                results['mpjpe'] = torch.mean(torch.stack(results['scalar/mpjpe'], dim=0))
                progress = (100 // display) * iter // (total_iters // display) + 1
                # print(100 * (epoch + 1) + progress - 1)
                write_tensorboard(writer, results, epoch+1, progress-1, 'train', args)

                end_time = time.time()
                time_used = (end_time - start_time) / 60.
                print('>>> [epoch {:2d}/ iter {:6d}] {:3d}%\n'
                      '    loss {:.4f}, tran {:.4f}, theta {:.4f}, joints3d {:.4f}, joints2d {:.4f} \n'
                      '    flow {:.4f}, delta_tran {:.4f}, mpjpe {:.4f} mm \n'
                      '    lr: {:.6f}, time used: {:.2f} mins, still need time for this epoch: {:.2f} mins.'
                      .format(epoch, iter, progress - 1, results['loss'], results['tran'], results['theta'],
                              results['joints3d'], results['joints2d'], results['flow'],
                              results['delta_tran'], 1000 * results['mpjpe'],
                              scheduler.get_last_lr()[0], time_used, (100 / progress - 1) * time_used))
        # break
        # '''

        # '''
        print('------------------------------------- test ------------------------------------')
        start_time = time.time()
        model.eval()  # dropout layers will not work in eval mode
        results = collections.defaultdict(list)
        results['faces'] = model.smpl.faces
        results['cam_intr'] = model.cam_intr
        with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
            for iter, data in enumerate(val_generator):
                # data: {events, flows, init_shape, theta, tran, joints2d, joints3d, info}
                for k in data.keys():
                    if k != 'info':
                        data[k] = data[k].to(device=device, dtype=dtype)

                if args.use_flow:
                    if args.use_flow_rgb:
                        out = model(data['events'], data['init_shape'], data['flows_rgb'], data['hidden_feats'])
                    else:
                        out = model(data['events'], data['init_shape'], data['flows'], data['hidden_feats'])
                else:
                    out = model(data['events'], data['init_shape'], None, data['hidden_feats'])

                loss_dict = compute_losses(out, data, mse_func, device, args)
                mpjpe = compute_mpjpe(out['joints3d'].detach(), data['joints3d'])  # [B, T, 24]
                pa_mpjpe = compute_pa_mpjpe(out['joints3d'].detach(), data['joints3d'])  # [B, T, 24]
                pel_mpjpe = compute_pelvis_mpjpe(out['joints3d'].detach(), data['joints3d'])  # [B, T, 24]
                loss = loss_dict['delta_tran'] + \
                       loss_dict['tran'] + \
                       loss_dict['theta'] + \
                       loss_dict['joints3d'] + \
                       loss_dict['joints2d'] + \
                       loss_dict['flow']

                # collect results
                results['scalar/delta_tran'].append(loss_dict['delta_tran'].detach())
                results['scalar/tran'].append(loss_dict['tran'].detach())
                results['scalar/theta'].append(loss_dict['theta'].detach())
                results['scalar/joints3d'].append(loss_dict['joints3d'].detach())
                results['scalar/joints2d'].append(loss_dict['joints2d'].detach())
                results['scalar/flow'].append(loss_dict['flow'].detach())
                results['scalar/loss'].append(loss.detach())
                results['scalar/mpjpe'].append(torch.mean(mpjpe.detach()))
                results['scalar/pa_mpjpe'].append(torch.mean(pa_mpjpe.detach()))
                results['scalar/pel_mpjpe'].append(torch.mean(pel_mpjpe.detach()))

                # if iter > 10:
                #     break
                # if iter % 2 == 0:
                if iter % 1000 == 0:
                    # print(100 * (epoch + 1) + iter // 1000)
                    results['info'] = (data['info'][0][0], data['info'][1][0])
                    results['verts'] = out['verts'][0].detach()
                    write_tensorboard(writer, results, epoch+1, iter//1000, 'test', args)

            results['delta_tran'] = torch.mean(torch.stack(results['scalar/delta_tran'], dim=0))
            results['tran'] = torch.mean(torch.stack(results['scalar/tran'], dim=0))
            results['theta'] = torch.mean(torch.stack(results['scalar/theta'], dim=0))
            results['joints3d'] = torch.mean(torch.stack(results['scalar/joints3d'], dim=0))
            results['joints2d'] = torch.mean(torch.stack(results['scalar/joints2d'], dim=0))
            results['flow'] = torch.mean(torch.stack(results['scalar/flow'], dim=0))
            results['loss'] = torch.mean(torch.stack(results['scalar/loss'], dim=0))
            results['mpjpe'] = torch.mean(torch.stack(results['scalar/mpjpe'], dim=0))
            results['pa_mpjpe'] = torch.mean(torch.stack(results['scalar/pa_mpjpe'], dim=0))
            results['pel_mpjpe'] = torch.mean(torch.stack(results['scalar/pel_mpjpe'], dim=0))
            end_time = time.time()
            time_used = (end_time - start_time) / 60.
            print('>>> loss {:.4f}, tran {:.4f}, theta {:.4f}, joints3d {:.4f}, joints2d {:.4f} \n'
                  '    flow {:.4f}, delta_tran {:.4f}, time used: {:.2f} mins \n'
                  '    mpjpe {:.4f}, pa_mpjpe {:.4f}, pel_mpjpe {:.4f} '
                  .format(results['loss'], results['tran'], results['theta'], results['joints3d'], results['joints2d'],
                          results['flow'], results['delta_tran'], time_used, 1000 * results['mpjpe'],
                          1000 * results['pa_mpjpe'], 1000 * results['pel_mpjpe']))

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, '%s/model.pkl' % save_dir)
            # torch.save(model.state_dict(), model_dir)

            if best_loss > results['loss']:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_dir)
                # torch.save(model.state_dict(), model_dir)
                best_loss = results['loss']
                print('>>> Model saved as {}... best loss {:.4f}'.format(model_dir, best_loss))
            # break
        # '''
        scheduler.step()
    writer.close()


def write_tensorboard(writer, results, epoch, progress, mode, args):
    action, sample_frames_idx = results['info']
    verts = results['verts'].cpu().numpy()  # [T+1, 6890, 3]
    cam_intr = results['cam_intr'].cpu().numpy()
    faces = results['faces']

    fullpics, render_imgs = [], []
    for i, frame_idx in enumerate(sample_frames_idx):
        img = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' % (args.data_dir, args.img_size, action, frame_idx))
        fullpics.append(img[:, :, 0:1])

        vert = verts[i]
        dist = np.abs(np.mean(vert, axis=0)[2])
        # render_img = (util.render_model(vert, faces, args.img_size, args.img_size, cam_intr, np.zeros([3]),
        #                                 np.zeros([3]), near=0.1, far=20 + dist, img=img) * 255).astype(np.uint8)
        render_img = util.render_model(vert, faces, args.img_size, args.img_size, cam_intr, np.zeros([3]),
                                       np.zeros([3]), near=0.1, far=20 + dist, img=img)
        render_imgs.append(render_img)

    fullpics = np.transpose(np.stack(fullpics, axis=0), [0, 3, 1, 2]) / 255.
    fullpics = np.concatenate([fullpics, fullpics, fullpics], axis=1)
    writer.add_images('%s/fullpic%06i' % (mode, epoch * 100 + progress), fullpics, 1, dataformats='NCHW')

    render_imgs = np.transpose(np.stack(render_imgs, axis=0), [0, 3, 1, 2])
    writer.add_images('%s/shape%06i' % (mode, epoch * 100 + progress), render_imgs, 1, dataformats='NCHW')


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
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='/home/shihao/data_event')
    parser.add_argument('--result_dir', type=str, default='/home/shihao/exp_track')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--smpl_dir', type=str, default='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--pin_memory', type=int, default=1)
    parser.add_argument('--use_amp', type=int, default=1)

    parser.add_argument('--events_input_channel', type=int, default=8)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--use_hmr_feats', type=int, default=1)
    parser.add_argument('--use_flow', type=int, default=1)
    parser.add_argument('--use_flow_rgb', type=int, default=0)
    parser.add_argument('--use_geodesic_loss', type=int, default=1)
    parser.add_argument('--vibe_regressor', type=int, default=0)
    parser.add_argument('--use_vibe_init', type=int, default=0)
    parser.add_argument('--use_hmr_init', type=int, default=0)

    parser.add_argument('--delta_tran_loss', type=float, default=0)
    parser.add_argument('--tran_loss', type=float, default=1)
    parser.add_argument('--theta_loss', type=float, default=10)
    parser.add_argument('--joints3d_loss', type=float, default=1)
    parser.add_argument('--joints2d_loss', type=float, default=10)
    parser.add_argument('--flow_loss', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=float, default=1)
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
