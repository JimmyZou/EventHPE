import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import joblib
import torch
from flow_net.flowlib import flow_to_image


class TrackingDataloader(Dataset):
    def __init__(
            self,
            data_dir='/home/shihao/data_event',
            max_steps=16,
            num_steps=8,
            skip=2,
            events_input_channel=8,
            img_size=256,
            mode='train',
            use_flow=True,
            use_flow_rgb=False,
            use_hmr_feats=False,
            use_vibe_init=False,
            use_hmr_init=False
    ):
        self.data_dir = data_dir
        self.events_input_channel = events_input_channel
        self.skip = skip
        self.max_steps = max_steps
        self.num_steps = num_steps
        self.img_size = img_size
        scale = self.img_size / 1280.
        self.cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
        self.use_hmr_feats = use_hmr_feats
        self.use_flow = use_flow
        self.use_flow_rgb = use_flow_rgb
        self.use_vibe_init = use_vibe_init
        self.use_hmr_init = use_hmr_init

        self.mode = mode
        if os.path.exists('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip)):
            self.all_clips = pickle.load(
                open('%s/%s_track%02i%02i.pkl' % (self.data_dir, self.mode, self.num_steps, self.skip), 'rb'))
        else:
            self.all_clips = self.obtain_all_clips()

        if self.use_vibe_init:
            print('[VIBE init]')
            all_clips = []
            for (action, frame_idx) in self.all_clips:
                if os.path.exists('%s/vibe_results_%02i%02i/%s/fullpic%04i_vibe%02i.pkl' %
                                  (self.data_dir, self.num_steps, self.skip, action, frame_idx, self.num_steps)):
                    all_clips.append((action, frame_idx))
                else:
                    print('[vibe not exist] %s %i' % (action, frame_idx))
            self.all_clips = all_clips

        if self.use_hmr_init:
            print('[hmr init]')
            all_clips = []
            for (action, frame_idx) in self.all_clips:
                if os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                    all_clips.append((action, frame_idx))
                else:
                    print('[hmr not exist] %s %i' % (action, frame_idx))
            self.all_clips = all_clips

        print('[%s] %i clips, track%02i%02i.pkl' % (self.mode, len(self.all_clips), self.num_steps, self.skip))

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        action, frame_idx = self.all_clips[idx]
        # TODO debug
        # action = 'subject02_group1_time1'
        # frame_idx = 1138
        if self.mode == 'train':
            next_frames_idx = self.skip * np.sort(np.random.choice(
                np.arange(1, self.max_steps+1), self.num_steps, replace=False))
        else:
            # test
            next_frames_idx = self.skip * np.arange(1, self.num_steps+1)

        sample_frames_idx = np.append(frame_idx, frame_idx + next_frames_idx)
        # print(sample_frames_idx)

        if self.use_vibe_init:
            _, _, _params, _tran = joblib.load(
                '%s/vibe_results_%02i%02i/%s/fullpic%04i_vibe%02i.pkl' %
                (self.data_dir, self.num_steps, self.skip, action, frame_idx, self.num_steps))
            theta = _params[0:1, 3:75]
            beta = _params[0:1, 75:]
            tran = _tran[0:1, :]
            init_shape = np.concatenate([tran, theta, beta], axis=1)
        elif self.use_hmr_init:
            _, _, _params, _tran, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx))
            theta = np.expand_dims(_params[3:75], axis=0)
            beta = np.expand_dims(_params[75:], axis=0)
            tran = _tran
            init_shape = np.concatenate([tran, theta, beta], axis=1)
        else:
            beta, theta, tran, _, _ = joblib.load('%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, frame_idx))
            init_shape = np.concatenate([tran, theta, beta], axis=1)

        if self.use_hmr_feats:
            _, _, _, _, hmr_feats = joblib.load(
                '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx))  # [2048]
        else:
            hmr_feats = np.zeros([2048])

        events, flows, flows_rgb, theta_list, tran_list, joints2d_list, joints3d_list = [], [], [], [], [], [], []
        for i in range(self.num_steps):
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
                if self.use_flow_rgb:
                    single_flows_rgb = np.transpose(
                        flow_to_image(np.transpose(single_flows, [1, 2, 0])), [2, 0, 1]) / 255.
                else:
                    single_flows_rgb = np.array([0])
            else:
                single_flows = np.array([0])
                single_flows_rgb = np.array([0])
            flows.append(single_flows)
            flows_rgb.append(single_flows_rgb)

            # single frame pose
            beta, theta, tran, joints3d, joints2d = joblib.load(
                '%s/pose_events/%s/pose%04i.pkl' % (self.data_dir, action, end_idx))
            theta_list.append(theta[0])
            tran_list.append(tran[0])
            joints2d_list.append(joints2d)
            joints3d_list.append(joints3d)

        events = np.stack(events, axis=0)  # [T, H, W, 8]
        flows = np.stack(flows, axis=0)  # [T, 2/3, H, W]
        theta_list = np.stack(theta_list, axis=0)  # [T, 72]
        tran_list = np.expand_dims(np.stack(tran_list, axis=0), axis=1)  # [T, 1, 3] in meter
        # [T, 24, 2] drop d, normalize to 0-1
        joints2d_list = np.stack(joints2d_list, axis=0)[:, :, 0:2] / self.img_size
        joints3d_list = np.stack(joints3d_list, axis=0)  # [T, 24, 3] added trans

        one_sample = {}
        one_sample['events'] = torch.from_numpy(np.transpose(events, [0, 3, 1, 2])).float()  # [T, 8, H, W]
        one_sample['flows'] = torch.from_numpy(flows).float()  # [T, 2, H, W]
        one_sample['flows_rgb'] = torch.from_numpy(flows).float()  # [T, 3, H, W]
        one_sample['init_shape'] = torch.from_numpy(init_shape).float()  # [1, 85]
        one_sample['hidden_feats'] = torch.from_numpy(hmr_feats).float()  # [2048]
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
            for i in range(len(frame_indices) - self.max_steps * self.skip):
                frame_idx = frame_indices[i]
                end_frame_idx = frame_idx + self.max_steps * self.skip
                if not os.path.exists('%s/pred_flow_events_%i/%s/flow%04i.pkl' %
                                      (self.data_dir, self.img_size, action, end_frame_idx)):
                    # print('flow %i not exists for %s-%i' % (end_frame_idx, action, frame_idx))
                    continue
                if not os.path.exists(
                        '%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (self.data_dir, action, frame_idx)):
                    continue
                if end_frame_idx == frame_indices[i + self.max_steps * self.skip]:
                    # action, frame_idx
                    all_clips.append((action, frame_idx))

        pickle.dump(all_clips, open('%s/%s_track%02i%02i.pkl' %
                                    (self.data_dir, self.mode, self.num_steps, self.skip), 'wb'))
        return all_clips

    def visualize(self, idx):
        sample = self.__getitem__(idx)
        action, sample_frames_idx = sample['info']
        events = np.transpose(sample['events'].numpy(), [0, 2, 3, 1])
        flows = np.transpose(sample['flows'].numpy(), [0, 2, 3, 1])
        joints3d = sample['joints3d'].numpy()
        joints2d = sample['joints2d'].numpy() * self.img_size

        # '''
        from event_pose_estimation.SMPL import SMPL
        model_dir = '../smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        device = torch.device('cpu')
        smpl_male = SMPL(model_dir, 1).to(device)

        import event_pose_estimation.utils as util
        import matplotlib.pyplot as plt
        from flow_net.flowlib import flow_to_image

        for t in range(self.num_steps // 2):
            plt.figure(figsize=(5 * 4, 5 * 4))

            # fullpic
            prev_fullpic = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' %
                                      (self.data_dir, self.img_size, action, sample_frames_idx[t]))
            curr_fullpic = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' %
                                      (self.data_dir, self.img_size, action, sample_frames_idx[t+1]))
            plt.subplot(4, 4, 1)
            plt.imshow(prev_fullpic[:, :, 0], cmap='gray')
            plt.axis('off')
            plt.title('%s-%04i' % (action, sample_frames_idx[t]))

            plt.subplot(4, 4, 2)
            plt.imshow(curr_fullpic[:, :, 0], cmap='gray')
            plt.axis('off')
            plt.title('%s-%04i' % (action, sample_frames_idx[t+1]))

            for i in range(self.num_steps):
                plt.subplot(4, 4, 3+i)
                plt.imshow(events[t, :, :, i], cmap='gray')
                plt.axis('off')

            beta = sample['init_shape'][:, -10:]
            theta = sample['theta'][t:t+1, :]
            verts, _, _ = smpl_male(beta, theta, get_skin=True)
            verts = (verts[0] + sample['tran'][t, :, :]).numpy()

            faces = smpl_male.faces
            dist = np.abs(np.mean(verts, axis=0)[2])
            render_img = (util.render_model(verts, faces, 256, 256, self.cam_intr, np.zeros([3]),
                          np.zeros([3]), near=0.1, far=20 + dist, img=curr_fullpic) * 255).astype(np.uint8)

            plt.subplot(4, 4, 11)
            plt.imshow(render_img)
            plt.axis('off')

            # joint2d and 3d
            img = curr_fullpic.copy()
            proj_joints2d = util.projection(joints3d[t], self.cam_intr, True)
            for point in proj_joints2d.astype(np.int64):
                cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), -1)
            plt.subplot(4, 4, 12)
            plt.imshow(img)
            plt.axis('off')
            plt.title('joints3d')

            img = curr_fullpic.copy()
            for point in joints2d[t].astype(np.int64):
                cv2.circle(img, (point[0], point[1]), 1, (255, 0, 0), -1)
            plt.subplot(4, 4, 13)
            plt.imshow(img)
            plt.axis('off')
            plt.title('joints2d')

            flow_rgb = flow_to_image(flows[t])
            plt.subplot(4, 4, 14)
            plt.imshow(flow_rgb)
            plt.axis('off')
            plt.title('flow')

            plt.show()
        # '''


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    data_train = TrackingDataloader(
        data_dir='/home/shihao/data_event_out',
        max_steps=16,
        num_steps=8,
        skip=2,
        events_input_channel=8,
        img_size=256,
        mode='train',
        use_hmr_feats=True
    )
    # sample = data_train[10000]
    data_train.visualize(20000)
    # print()
    # for k, v in sample.items():
    #     if k is not 'info':
    #         print(k, v.size())

    # data_test = TrackingDataloader(
    #     data_dir='/home/shihao/data_event',
    #     max_steps=16,
    #     num_steps=8,
    #     skip=2,
    #     events_input_channel=8,
    #     img_size=256,
    #     mode='train',
    #     use_flow=True,
    #     use_flow_rgb=False,
    #     use_hmr_feats=False,
    #     use_vibe_init=False,
    #     use_hmr_init=True,
    # )
    # # data_test.visualize(30000)
    # sample = data_test[30000]
    # for k, v in sample.items():
    #     if k != 'info':
    #         print(k, v.size())

