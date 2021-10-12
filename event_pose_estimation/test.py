import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import time
import joblib
import cv2
import pickle
import sys
sys.path.append('../')
from event_pose_estimation.model import EventTrackNet
from event_pose_estimation.SMPL import SMPL
from event_pose_estimation.geometry import rotation_matrix_to_angle_axis, batch_compute_similarity_transform_torch
from event_pose_estimation.loss_funcs import compute_mpjpe, compute_pa_mpjpe, compute_pelvis_mpjpe, \
    compute_pck, compute_pck_head, compute_pck_torso
import collections
import numpy as np


class TrackingTestDataloader(Dataset):
    def __init__(
            self,
            data_dir='/home/shihao/data_event',
            train_steps=8,
            test_steps=8,
            skip=2,
            events_input_channel=8,
            img_size=256,
            mode='test',
            use_flow=True,
            use_hmr_feats=False,
            target_action=None,
            use_vibe_init=False,
            use_hmr_init=False,
    ):
        self.data_dir = data_dir
        self.events_input_channel = events_input_channel
        self.skip = skip
        self.train_steps = train_steps
        self.test_steps = test_steps
        self.img_size = img_size
        scale = self.img_size / 1280.
        self.cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
        self.use_hmr_feats = use_hmr_feats
        self.use_flow = use_flow
        self.use_vibe_init = use_vibe_init
        self.use_hmr_init = use_hmr_init

        self.mode = mode
        if os.path.exists('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.test_steps, self.skip)):
            clips = pickle.load(
                open('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.test_steps, self.skip), 'rb'))
        else:
            clips = self.obtain_all_clips()

        if target_action:
            self.all_clips = []
            for (action, frame_idx) in clips:
                if target_action in action:
                    self.all_clips.append((action, frame_idx))
        else:
            self.all_clips = clips
        print('[%s] %i clips, test_track%02i%02i.pkl' % (self.mode, len(self.all_clips), self.test_steps, self.skip))

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        action, frame_idx = self.all_clips[idx]
        # action, frame_idx = 'subject02_group2_time1', 710

        # test
        next_frames_idx = self.skip * np.arange(1, self.test_steps+1)
        sample_frames_idx = np.append(frame_idx, frame_idx + next_frames_idx)
        # print(sample_frames_idx)

        # beta, theta, tran, _, _ = joblib.load('%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, frame_idx))
        # init_shape = np.concatenate([tran, theta, beta], axis=1)

        hmr_feats, init_shapes = [], []
        for i in range(0, self.test_steps, self.train_steps):
            if self.use_vibe_init:
                fname = '%s/vibe_results_%02i%02i/%s/fullpic%04i_vibe%02i.pkl' % \
                        (self.data_dir, self.test_steps, self.skip, action, frame_idx, self.test_steps)
                _, _, _params, _tran = joblib.load(fname)
                theta = _params[0:1, 3:75]
                beta = _params[0:1, 75:]
                tran = _tran[0:1, :]
                init_shape = np.concatenate([tran, theta, beta], axis=1)
            elif self.use_hmr_init:
                _, _, _params, _tran, _ = \
                    joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, sample_frames_idx[i]))
                theta = np.expand_dims(_params[3:75], axis=0)
                beta = np.expand_dims(_params[75:], axis=0)
                tran = _tran
                init_shape = np.concatenate([tran, theta, beta], axis=1)
            else:
                beta, theta, tran, _, _ = joblib.load('%s/pose_events/%s/pose%04i.pkl' %
                                                      (self.data_dir, action, sample_frames_idx[i]))
                init_shape = np.concatenate([tran, theta, beta], axis=1)
            init_shapes.append(init_shape)

            if self.use_hmr_feats:
                _, _, _, _, hmr_feat = joblib.load(
                    '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, sample_frames_idx[i]))  # [2048]
                hmr_feats.append(hmr_feat)
            else:
                hmr_feats.append(np.zeros([2048]))
        hmr_feats = np.stack(hmr_feats, axis=0)  # [test_steps // tran_steps, 2048]
        init_shapes = np.stack(init_shapes, axis=0)

        events, flows, theta_list, tran_list, joints2d_list, joints3d_list = [], [], [], [], [], []
        for i in range(self.test_steps):
            start_idx = sample_frames_idx[i]
            end_idx = sample_frames_idx[i+1]
            # print('frame %i - %i' % (start_idx, end_idx))

            # single step events frame
            single_events_frame = []
            for j in range(start_idx, end_idx):
                single_events_frame.append(cv2.imread(
                    '%s/events_%i/%s/event%04i.png' % (self.data_dir, self.img_size, action, j), -1))
            single_events_frame = np.concatenate(single_events_frame, axis=2).astype(np.float32)  # [H, W, C]
            # aggregate the events frame to get 8 channel
            if single_events_frame.shape[2] > self.events_input_channel:
                skip = single_events_frame.shape[2] // self.events_input_channel
                idx1 = skip * np.arange(self.events_input_channel)
                idx2 = idx1 + skip
                idx2[-1] = max(idx2[-1], single_events_frame.shape[2])
                single_events_frame = np.stack(
                    [(np.sum(single_events_frame[:, :, c1:c2], axis=2) > 0) for (c1, c2) in zip(idx1, idx2)], axis=2)
            events.append(single_events_frame)

            if self.use_flow:
                # single step flow
                single_flows = [joblib.load(
                    '%s/pred_flow_events_%i/%s/flow%04i.pkl' % (self.data_dir, self.img_size, action, j))
                    for j in range(start_idx, end_idx, self.skip)]  # flow is predicted with skip=2
                # single flow is saved as int16 to save disk memory, [T, C, H, W]
                single_flows = np.stack(single_flows, axis=0).astype(np.float32) / 100
                single_flows = np.sum(single_flows, axis=0)
            else:
                single_flows = np.array([0])
            flows.append(single_flows)

            # single frame pose
            beta, theta, tran, joints3d, joints2d = joblib.load(
                '%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, end_idx))
            theta_list.append(theta[0])
            tran_list.append(tran[0])
            joints2d_list.append(joints2d)
            joints3d_list.append(joints3d)

        events = np.stack(events, axis=0)  # [T, H, W, 8]
        flows = np.stack(flows, axis=0)  # [T, 2, H, W]
        theta_list = np.stack(theta_list, axis=0)  # [T, 72]
        tran_list = np.expand_dims(np.stack(tran_list, axis=0), axis=1)  # [T, 1, 3] in meter
        # [T, 24, 2] drop d, normalize to 0-1
        joints2d_list = np.stack(joints2d_list, axis=0)[:, :, 0:2] / self.img_size
        joints3d_list = np.stack(joints3d_list, axis=0)  # [T, 24, 3] added trans

        one_sample = {}
        one_sample['events'] = torch.from_numpy(np.transpose(events, [0, 3, 1, 2])).float()  # [T, 8, H, W]
        one_sample['flows'] = torch.from_numpy(flows).float()  # [T, 2, H, W]
        one_sample['init_shape'] = torch.from_numpy(init_shapes).float()  # [1, 85]
        one_sample['hidden_feats'] = torch.from_numpy(hmr_feats).float()  # [test_steps // tran_steps, 2048]
        one_sample['theta'] = torch.from_numpy(theta_list).float()  # [T, 72]
        one_sample['tran'] = torch.from_numpy(tran_list).float()  # [T, 1, 3]
        one_sample['joints2d'] = torch.from_numpy(joints2d_list).float()  # [T, 24, 2]
        one_sample['joints3d'] = torch.from_numpy(joints3d_list).float()  # [T, 24, 3]
        one_sample['info'] = [action, sample_frames_idx]
        return one_sample

    def obtain_all_clips(self):
        all_clips = []
        tmp = sorted(os.listdir('%s/pose_events' % self.data_dir))
        action_names = []
        for action in tmp:
            subject = action.split('_')[0]
            if self.mode == 'test':
                if subject in ['subject01', 'subject02', 'subject07']:
                    action_names.append(action)
            else:
                if subject not in ['subject01', 'subject02', 'subject07']:
                    action_names.append(action)

        for action in action_names:
            if not os.path.exists('%s/pose_events/%s/pose_info.pkl' % (self.data_dir, action)):
                print('[warning] not exsit %s/pose_events/%s/pose_info.pkl' % (self.data_dir, action))
                continue

            frame_indices = joblib.load('%s/pose_events/%s/pose_info.pkl' % (self.data_dir, action))
            for i in range(len(frame_indices) - self.test_steps * self.skip):
                frame_idx = frame_indices[i]
                end_frame_idx = frame_idx + self.test_steps * self.skip
                if not os.path.exists('%s/pred_flow_events_%i/%s/flow%04i.pkl' %
                                      (self.data_dir, self.img_size, action, end_frame_idx)):
                    # print('flow %i not exists for %s-%i' % (end_frame_idx, action, frame_idx))
                    continue
                if not os.path.exists(
                        '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                    continue
                if end_frame_idx == frame_indices[i + self.test_steps * self.skip]:
                    # action, frame_idx
                    all_clips.append((action, frame_idx))

        pickle.dump(all_clips, open('%s/%s_track%02i%02i.pkl' %
                                    (self.data_dir, self.mode, self.test_steps, self.skip), 'wb'))
        return all_clips


def test_simple_instance(args):
    # GPU or CPU configuration
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    device = torch.device("cpu")
    dtype = torch.float32

    # set dataset
    dataset_test = TrackingTestDataloader(
        data_dir=args.data_dir,
        train_steps=args.train_steps,  # corresponds to num_steps
        test_steps=args.test_steps,
        skip=args.skip,
        events_input_channel=args.events_input_channel,
        img_size=args.img_size,
        mode='test',
        use_hmr_feats=args.use_hmr_feats
    )
    test_generator = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker,
        pin_memory=False
    )

    # set model
    model = EventTrackNet(
        events_input_channel=args.events_input_channel,
        smpl_dir=args.smpl_dir,
        batch_size=args.model_batchsize,
        num_steps=args.train_steps,
        n_layers=args.rnn_layers,
        hidden_size=2048,
        bidirectional=False,
        add_linear=False,
        use_residual=True,
        pose_dim=24*6,
        cam_intr=torch.from_numpy(dataset_test.cam_intr).float()
    )

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # set tensorboard
    if args.model_dir is not None:
        print('[model dir] model loaded from %s' % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.smpl = SMPL(args.smpl_dir, args.batch_size * args.train_steps).to(device=device)
    else:
        raise ValueError('Cannot find trained model %s.' % args.model_dir)

    # check path
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('------------------------------------- test ------------------------------------')
    start_time = time.time()
    model.eval()  # dropout layers will not work in eval mode
    results = collections.defaultdict(list)
    results['faces'] = model.smpl.faces
    results['cam_intr'] = model.cam_intr
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for iter, data in enumerate(test_generator):
            # data: {events, flows, init_shape, theta, tran, joints2d, joints3d, info}
            for k in data.keys():
                if k != 'info':
                    data[k] = data[k].to(device=device, dtype=dtype)

            B, T = data['events'].size()[0], data['events'].size()[1]
            trans, verts, joints3d, joints2d = [], [], [], []
            init_shape = data['init_shape'][:, 0]
            for i in range(args.test_steps // args.train_steps):
                start_idx = i * args.train_steps
                end_idx = min((i + 1) * args.train_steps, args.test_steps)

                out = model(
                    data['events'][:, start_idx:end_idx],
                    # data['init_shape'][:, i],
                    init_shape,
                    data['flows'][:, start_idx:end_idx],
                    data['hidden_feats'][:, i]
                )

                # get new init shape
                tran = out['tran'][:, -1, :, :]  # [B, 1, 3]
                theta = rotation_matrix_to_angle_axis(out['pred_rotmats'][:, -1, :, :, :].reshape(B*24, 3, 3))
                theta = theta.view(B, 24, 3).reshape(B, 1, 72)
                beta = data['init_shape'][:, 0, :, 75:85]  # [B, 1, 10]
                init_shape = torch.cat([tran, theta, beta], dim=2)  # [B, 1, 85]

                # collect result
                trans.append(out['tran'])  # [B, T, 1, 3]
                verts.append(out['verts'][:, min(start_idx, 1):])  # [B, T+1, 6890, 3]
                joints3d.append(out['joints3d'])  # [B, T, 24, 3]
                joints2d.append(out['joints2d'])  # [B, T, 24, 3]

            trans = torch.cat(trans, dim=1)
            verts = torch.cat(verts, dim=1)
            joints3d = torch.cat(joints3d, dim=1)
            joints2d = torch.cat(joints2d, dim=1)
            # print(trans.size(), verts.size(), joints3d.size(), joints2d.size())

            mpjpe = compute_mpjpe(joints3d, data['joints3d'])  # [B, T, 24]
            pa_mpjpe = compute_pa_mpjpe(joints3d, data['joints3d'])  # [B, T, 24]
            pel_mpjpe = compute_pelvis_mpjpe(joints3d, data['joints3d'])  # [B, T, 24]
            pck = compute_pck(joints3d, data['joints3d'])  # [B, T, 24]

            # [B*T, 3], [B*T, 3, 3], [B*T, 3, 1]
            _, s, R, t = batch_compute_similarity_transform_torch(
                joints3d.view(-1, 24, 3), data['joints3d'].view(-1, 24, 3), True)
            # print(s.size(), R.size(), t.size())
            pa_verts = s.unsqueeze(-1).unsqueeze(-1) * R.bmm(verts[:, 1:].reshape(B * T, 6890, 3).permute(0, 2, 1)) + t
            pa_verts = pa_verts.permute(0, 2, 1).view(B, T, 6890, 3)
            pa_verts = torch.cat([verts[:, 0:1], pa_verts], dim=1)

            # collect results
            results['scalar/mpjpe'].append(mpjpe.detach())
            results['scalar/pa_mpjpe'].append(pa_mpjpe.detach())
            results['scalar/pel_mpjpe'].append(pel_mpjpe.detach())
            results['scalar/pck'].append(pck.float().detach())

            for i in range(B):
                action, sampled_frames_idx = data['info'][0][i], data['info'][1][i]
                frame_idx = sampled_frames_idx[0]
                # print(action, frame_idx)
                if not os.path.exists('%s/%s' % (save_dir, action)):
                    os.mkdir('%s/%s' % (save_dir, action))

                _tran = trans[i].cpu().numpy()
                _verts = verts[i].cpu().numpy()
                _pa_verts = pa_verts[i].cpu().numpy()
                _joints3d = joints3d[i].cpu().numpy()
                _joints2d = joints2d[i].cpu().numpy()
                _mpjpe = mpjpe[i].cpu().numpy()
                _pa_mpjpe = pa_mpjpe[i].cpu().numpy()
                _pck = pck[i].cpu().numpy()
                fname = '%s/%s/frame%04i_%02i%02i%02i.pkl' % \
                        (save_dir, action, frame_idx, args.test_steps, args.train_steps, args.skip)
                joblib.dump([_verts, _pa_verts, _tran, _joints3d, _joints2d, _mpjpe, _pa_mpjpe, _pck], fname, compress=3)

            if iter > 1:
                break

        results['mpjpe'] = torch.mean(torch.cat(results['scalar/mpjpe'], dim=0), dim=(0, 2))
        results['pa_mpjpe'] = torch.mean(torch.cat(results['scalar/pa_mpjpe'], dim=0), dim=(0, 2))
        results['pel_mpjpe'] = torch.mean(torch.cat(results['scalar/pel_mpjpe'], dim=0), dim=(0, 2))
        results['pck'] = torch.mean(torch.cat(results['scalar/pck'], dim=0), dim=(0, 2))

        end_time = time.time()
        time_used = (end_time - start_time) / 60.
        print('>>> time used: {:.2f} mins \n'
              '    mpjpe {}\n'
              '    pa_mpjpe {}\n'
              '    pel_mpjpe {}\n'
              '    pck {}'
              .format(time_used, 1000 * results['mpjpe'], 1000 * results['pa_mpjpe'],
                      1000 * results['pel_mpjpe'], results['pck']))

        # '''


def test_whole_set(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # set dataset
    dataset_test = TrackingTestDataloader(
        data_dir=args.data_dir,
        train_steps=args.test_steps,  # corresponds to num_steps
        test_steps=args.test_steps,
        skip=args.skip,
        events_input_channel=args.events_input_channel,
        img_size=args.img_size,
        mode='test',
        use_flow=args.use_flow,
        use_hmr_feats=args.use_hmr_feats,
        target_action=args.target_action,
        use_vibe_init=args.use_vibe_init,
        use_hmr_init=args.use_hmr_init,
    )
    test_generator = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_worker,
        pin_memory=False
    )

    # set model
    model = EventTrackNet(
        events_input_channel=args.events_input_channel,
        smpl_dir=args.smpl_dir,
        batch_size=args.model_batchsize,
        num_steps=args.train_steps,
        n_layers=args.rnn_layers,
        hidden_size=2048,
        bidirectional=False,
        add_linear=False,
        use_residual=True,
        pose_dim=24 * 6,
        use_flow=args.use_flow,
        vibe_regressor=args.vibe_regressor,
        cam_intr=torch.from_numpy(dataset_test.cam_intr).float()
    )
    model = model.to(device=device)  # move the model parameters to CPU/GPU

    # set tensorboard
    if args.model_dir is not None:
        print('[model dir] model loaded from %s' % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.smpl = SMPL(args.smpl_dir, args.batch_size * args.train_steps).to(device=device)
    else:
        raise ValueError('Cannot find trained model %s.' % args.model_dir)

    # check path
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('------------------------------------- test ------------------------------------')
    start_time = time.time()
    model.eval()  # dropout layers will not work in eval mode
    results = collections.defaultdict(list)
    results['faces'] = model.smpl.faces
    results['cam_intr'] = model.cam_intr
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for iter, data in enumerate(test_generator):
            # data: {events, flows, init_shape, theta, tran, joints2d, joints3d, info}
            for k in data.keys():
                if k != 'info':
                    data[k] = data[k].to(device=device, dtype=dtype)

            B, T = data['events'].size()[0], data['events'].size()[1]
            init_shape = data['init_shape'][:, 0]

            if args.use_flow:
                out = model(data['events'], init_shape, data['flows'], data['hidden_feats'][:, 0])
            else:
                out = model(data['events'], init_shape, None, data['hidden_feats'][:, 0])

            mpjpe = compute_mpjpe(out['joints3d'], data['joints3d'])  # [B, T, 24]
            pa_mpjpe = compute_pa_mpjpe(out['joints3d'], data['joints3d'])  # [B, T, 24]
            pel_mpjpe = compute_pelvis_mpjpe(out['joints3d'], data['joints3d'])  # [B, T, 24]
            pck = compute_pck(out['joints3d'], data['joints3d'])  # [B, T, 24]
            pck_head = compute_pck_head(out['joints3d'], data['joints3d'])  # [B, T, 24]
            pck_torso = compute_pck_torso(out['joints3d'], data['joints3d'])  # [B, T, 24]

            # [B*T, 3], [B*T, 3, 3], [B*T, 3, 1]
            _, s, R, t = batch_compute_similarity_transform_torch(
                out['joints3d'].view(-1, 24, 3), data['joints3d'].view(-1, 24, 3), True)
            # print(s.size(), R.size(), t.size())
            s = s.unsqueeze(-1).unsqueeze(-1)
            pa_verts = s * R.bmm(out['verts'][:, 1:].reshape(B * T, 6890, 3).permute(0, 2, 1)) + t
            pa_verts = pa_verts.permute(0, 2, 1).view(B, T, 6890, 3)
            pa_verts = torch.cat([out['verts'][:, 0:1], pa_verts], dim=1)  # [B, T+1, 6890, 3]

            # get target vertex
            beta = init_shape[:, :, 75:85].repeat(1, T, 1)  # [B, T, 10]
            target_verts, target_joints3d, _ = model.smpl(
                beta=beta.view(-1, 10),
                theta=data['theta'].view(-1, 72),
                get_skin=True)
            target_verts = target_verts.view(B, T, target_verts.size(1), target_verts.size(2)) + data['tran']
            target_verts = torch.cat([out['verts'][:, 0:1], target_verts], dim=1)  # [B, T+1, 6890, 3]
            # target_joints3d = target_joints3d.view(B, T, target_joints3d.size(1), target_joints3d.size(2)) + data['tran']

            # print(data['info'])
            # print(1000 * torch.mean(torch.sqrt(torch.sum((target_verts[:, 1:] - pa_verts[:, 1:]) ** 2, dim=-1))))
            # print(torch.mean(pck.detach().float()))

            pve = torch.mean(torch.sqrt(torch.sum((target_verts[:, 1:] - pa_verts[:, 1:]) ** 2, dim=-1)), dim=-1)  # [B, T]

            # collect results
            results['scalar/mpjpe'].append(mpjpe.detach())
            results['scalar/pa_mpjpe'].append(pa_mpjpe.detach())
            results['scalar/pel_mpjpe'].append(pel_mpjpe.detach())
            results['scalar/pck'].append(pck.detach().float())
            results['scalar/pck_head'].append(pck_head.detach().float())
            results['scalar/pck_torso'].append(pck_torso.detach().float())
            results['scalar/pve'].append(pve.detach())

            # collect result
            trans = out['tran']  # [B, T, 1, 3]
            verts = out['verts']  # [B, T+1, 6890, 3]
            joints3d = out['joints3d']  # [B, T, 24, 3]
            joints2d = out['joints2d']  # [B, T, 24, 3]

            if args.save_results:
                for i in range(B):
                    action, sampled_frames_idx = data['info'][0][i], data['info'][1][i]
                    frame_idx = sampled_frames_idx[0]
                    # print(action, frame_idx)
                    if not os.path.exists('%s/%s' % (save_dir, action)):
                        os.mkdir('%s/%s' % (save_dir, action))

                    _tran = trans[i].cpu().numpy()
                    _verts = verts[i].cpu().numpy()
                    _pa_verts = pa_verts[i].cpu().numpy()
                    _joints3d = joints3d[i].cpu().numpy()
                    _joints2d = joints2d[i].cpu().numpy()
                    _mpjpe = mpjpe[i].cpu().numpy()
                    _pa_mpjpe = pa_mpjpe[i].cpu().numpy()
                    _pck = pck[i].cpu().numpy()
                    fname = '%s/%s/frame%04i_%02i%02i%02i.pkl' % \
                            (save_dir, action, frame_idx, args.test_steps, args.train_steps, args.skip)
                    joblib.dump([_verts, _pa_verts, _tran, _joints3d, _joints2d, _mpjpe, _pa_mpjpe, _pck],
                                fname, compress=3)

            # if iter > 5:
            #     break

        results['mpjpe'] = torch.mean(torch.cat(results['scalar/mpjpe'], dim=0), dim=(0, 1, 2))
        results['pa_mpjpe'] = torch.mean(torch.cat(results['scalar/pa_mpjpe'], dim=0), dim=(0, 1, 2))
        results['pel_mpjpe'] = torch.mean(torch.cat(results['scalar/pel_mpjpe'], dim=0), dim=(0, 1, 2))
        results['pck'] = torch.mean(torch.cat(results['scalar/pck'], dim=0), dim=(0, 1, 2))
        results['pck_head'] = torch.mean(torch.cat(results['scalar/pck_head'], dim=0), dim=(0, 1, 2))
        results['pck_torso'] = torch.mean(torch.cat(results['scalar/pck_torso'], dim=0), dim=(0, 1, 2))
        results['pve'] = torch.mean(torch.cat(results['scalar/pve'], dim=0))

        end_time = time.time()
        time_used = (end_time - start_time) / 60.
        print('>>> time used: {:.2f} mins \n'
              '    mpjpe {}\n'
              '    pa_mpjpe {}\n'
              '    pel_mpjpe {}\n'
              '    pck {}\n'
              '    pck_head {}\n'
              '    pck_torso {}\n'
              '    pve {}\n'
              .format(time_used, 1000 * results['mpjpe'], 1000 * results['pa_mpjpe'],
                      1000 * results['pel_mpjpe'], results['pck'], results['pck_head'],
                      results['pck_torso'], 1000 * results['pve']))
        # '''


def get_args():
    def print_args(args):
        """ Prints the argparse argmuments applied
        Args:
         ll - args = parser.parse_args()
        """
        _args = vars(args)
        max_length = max([len(k) for k, _ in _args.items()])
        for k, v in _args.items():
            print(' ' * (max_length - len(k)) + k + ': ' + str(v))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='/home/shihao/data_event_out')
    parser.add_argument('--save_dir', type=str, default='/home/shihao/data_event_out/ours_vibe_init')
    parser.add_argument('--model_dir', type=str, default='/home/shihao/data_event_out/model/ours_gt.pkl')

    parser.add_argument('--events_input_channel', type=int, default=8)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--test_steps', type=int, default=8)
    parser.add_argument('--train_steps', type=int, default=8)
    parser.add_argument('--skip', type=int, default=2)

    parser.add_argument('--model_batchsize', type=int, default=16)  # batch size used when training the model
    parser.add_argument('--use_hmr_feats', type=int, default=1)
    parser.add_argument('--use_flow', type=int, default=1)
    parser.add_argument('--vibe_regressor', type=int, default=0)

    parser.add_argument('--target_action', type=str, default='subject02_group1_time1')
    parser.add_argument('--save_results', type=int, default=0)

    parser.add_argument('--use_vibe_init', type=int, default=1)
    parser.add_argument('--use_hmr_init', type=int, default=0)
    parser.add_argument('--smpl_dir', type=str, default='../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_worker', type=int, default=4)
    args = parser.parse_args()
    print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    # test_whole_set(args)
    test_simple_instance(args)

    # data = TrackingTestDataloader(
    #     data_dir='/home/shihao/data_event',
    #     train_steps=8,
    #     test_steps=8,
    #     skip=2,
    #     events_input_channel=8,
    #     img_size=256,
    #     mode='test',
    #     use_flow=True,
    #     use_hmr_feats=False,
    #     target_action=None,
    #     use_vibe_init=True
    # )
    # # print(data.all_clips)
    # sample = data[1000]
    # for k, v in sample.items():
    #     if k is not 'info':
    #         print(k, v.size())


if __name__ == '__main__':
    main()
