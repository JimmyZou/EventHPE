import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils.data import Dataset
import pickle
import joblib
import glob
import torch
from albumentations import ShiftScaleRotate, Compose


class FlowDataloader(Dataset):
    def __init__(self, data_dir='/home/shihao/data_event', input_channel=8, img_size=256, num_skip=4, skip=2,
                 max_num_events=60000, transform=None, mode='train', viz=False, source='events'):
        self.data_dir = data_dir
        self.transform = transform
        self.skip = skip
        self.input_channel = input_channel
        self.img_size = img_size
        self.mode = mode
        self.viz = viz
        self.source = source
        if self.mode == 'all':
            tmp1 = pickle.load(
                open('%s/test_flow%i%02i.pkl' % (self.data_dir, self.img_size, num_skip), 'rb'))
            tmp2 = pickle.load(
                open('%s/train_flow%i%02i.pkl' % (self.data_dir, self.img_size, num_skip), 'rb'))
            self.all_files = tmp1 + tmp2
        else:
            if os.path.exists('%s/%s_flow%i%02i.pkl' % (self.data_dir, self.mode, self.img_size, num_skip)):
                self.all_files = pickle.load(
                    open('%s/%s_flow%i%02i.pkl' % (self.data_dir, self.mode, self.img_size, num_skip), 'rb'))
            else:
                self.all_files = self.obtain_all_files(num_skip, skip, max_num_events)
        print('Dataloder finished... %i samples...%s_flow%i%02i.pkl' %
              (len(self.all_files), self.mode, self.img_size, num_skip))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        action, frame_idx, prob = self.all_files[idx]
        one_sample = {}
        one_sample['info'] = '%s-%04i' % (action, frame_idx)

        if self.mode == 'train':
            next_frame_skip = np.random.choice(self.skip * np.arange(1, len(prob)+1), p=prob)
        else:
            next_frame_skip = self.skip
        # print(action, frame_idx, next_frame_skip, prob)

        if self.source == 'color':
            img1 = cv2.imread('%s/color_%i/%s/color%04i.jpg' %
                              (self.data_dir, self.img_size, action, frame_idx))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.imread('%s/color_%i/%s/color%04i.jpg' %
                              (self.data_dir, self.img_size, action, frame_idx+next_frame_skip))
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

            # self.visualize(fullpic1, fullpic2, events_frame, idx, next_frame_skip)
            if self.transform:
                img1, img2 = self.transform([img1, img2])
            if self.viz:
                self.visualize(img1, img2, np.array([0]), idx, next_frame_skip)

            # img1, img2, events_frame, next_frame_skip / self.skip
            one_sample['img1'] = torch.from_numpy(np.transpose(img1, [2, 0, 1])).float()
            one_sample['img2'] = torch.from_numpy(np.transpose(img2, [2, 0, 1])).float()
            one_sample['events_frame'] = torch.from_numpy(np.array([0])).float()
            one_sample['dt'] = torch.tensor([next_frame_skip / self.skip]).float()

        else:
            fullpic1 = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' %
                                  (self.data_dir, self.img_size, action, frame_idx))[:, :, 0:1]
            fullpic2 = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' %
                                  (self.data_dir, self.img_size, action, frame_idx+next_frame_skip))[:, :, 0:1]

            if self.source == 'events':
                events = []
                for i in range(next_frame_skip):
                    events.append(cv2.imread('%s/events_%i/%s/event%04i.png' %
                                             (self.data_dir, self.img_size, action, frame_idx+i), -1))
                events = np.concatenate(events, axis=2)
                gap = events.shape[2] // self.input_channel
                # print(gap, events.shape)
                events_frame = np.stack(
                    [(np.sum(events[:, :, i*gap:(i+1)*gap], axis=2) > 0).astype(np.float32)
                     for i in np.arange(0, self.input_channel)], axis=2)

                # self.visualize(fullpic1, fullpic2, events_frame, idx, next_frame_skip)
                if self.transform:
                    fullpic1, fullpic2, events_frame = self.transform([fullpic1, fullpic2, events_frame])
                if self.viz:
                    self.visualize(fullpic1, fullpic2, events_frame, idx, next_frame_skip)

                # img1, img2, events_frame, next_frame_skip / self.skip
                one_sample['img1'] = torch.from_numpy(np.transpose(fullpic1, [2, 0, 1])).float()
                one_sample['img2'] = torch.from_numpy(np.transpose(fullpic2, [2, 0, 1])).float()
                one_sample['events_frame'] = torch.from_numpy(np.transpose(events_frame, [2, 0, 1])).float()
                one_sample['dt'] = torch.tensor([next_frame_skip / self.skip]).float()
            else:
                if self.transform:
                    fullpic1, fullpic2 = self.transform([fullpic1, fullpic2])
                if self.viz:
                    self.visualize(fullpic1, fullpic2, np.array([0]), idx, next_frame_skip)

                # img1, img2, events_frame, next_frame_skip / self.skip
                one_sample['img1'] = torch.from_numpy(np.transpose(fullpic1, [2, 0, 1])).float()
                one_sample['img2'] = torch.from_numpy(np.transpose(fullpic2, [2, 0, 1])).float()
                one_sample['events_frame'] = torch.from_numpy(np.array([0])).float()
                one_sample['dt'] = torch.tensor([next_frame_skip / self.skip]).float()
        return one_sample

    def obtain_all_files(self, num_skip, skip, max_num_events):
        all_files = []
        tmp = sorted(os.listdir('%s/full_pic_%i' % (self.data_dir, self.img_size)))
        action_names = []
        for action in tmp:
            if self.mode == 'test':
                if 'subject02' in action:
                    action_names.append(action)
            else:
                if 'subject02' not in action:
                    action_names.append(action)

        for action in action_names:
            if not os.path.exists('%s/events_%i/%s/%s_info.pkl' % (self.data_dir, self.img_size, action, action)):
                print('[warning] not exsit %s/events_%i/%s/%s_info.pkl' %
                      (self.data_dir, self.img_size, action, action))
                continue

            indices, events_count = joblib.load('%s/events_%i/%s/%s_info.pkl' %
                                                (self.data_dir, self.img_size, action, action))
            for i in range(len(indices) - num_skip * skip):
                prob = []
                for j in range(num_skip):
                    prob.append(sum(events_count[i+j*skip:i+(j+1)*skip]))
                    if sum(prob) > max_num_events:
                        break
                prob = np.array(prob) / np.sum(prob)
                # action, frame_idx, probability of next frame
                all_files.append((action, indices[i], prob))

        pickle.dump(all_files, open('%s/%s_flow%i%02i.pkl' % (self.data_dir, self.mode, self.img_size, num_skip), 'wb'))
        return all_files

    def visualize(self, fullpic1, fullpic2, events_frame, idx, next_frame_skip):
        action, frame_idx, prob = self.all_files[idx]
        if self.source == 'events':
            n = events_frame.shape[2] + 2

            plt.figure(figsize=(2*n, 10))

            plt.subplot(2, n//2, 1)
            plt.imshow(fullpic1[:, :, 0], cmap='gray')
            # plt.axis('off')
            plt.title('%s_%04i' % (action, frame_idx), fontdict={'fontsize': 14})

            for i in range(n-2):
                plt.subplot(2, n//2, i+2)
                plt.imshow(events_frame[:, :, i], cmap='gray')
                plt.title('# events: %i' % np.sum(events_frame[:, :, i]), fontdict={'fontsize': 14})
                # plt.axis('off')

            plt.subplot(2, n//2, n)
            plt.imshow(fullpic2[:, :, 0], cmap='gray')
            # plt.axis('off')
            plt.title('%s_%04i' % (action, frame_idx+next_frame_skip), fontdict={'fontsize': 14})

            plt.show()
        elif self.source == 'fullpic':
            plt.figure(figsize=(8, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(fullpic1[:, :, 0], cmap='gray')
            # plt.axis('off')
            plt.title('%s_%04i' % (action, frame_idx), fontdict={'fontsize': 14})

            plt.subplot(1, 2, 2)
            plt.imshow(fullpic2[:, :, 0], cmap='gray')
            # plt.axis('off')
            plt.title('%s_%04i' % (action, frame_idx + next_frame_skip), fontdict={'fontsize': 14})

            plt.show()
        else:
            plt.figure(figsize=(8, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(fullpic1)
            # plt.axis('off')
            plt.title('%s_%04i' % (action, frame_idx), fontdict={'fontsize': 14})

            plt.subplot(1, 2, 2)
            plt.imshow(fullpic2, cmap='gray')
            # plt.axis('off')
            plt.title('%s_%04i' % (action, frame_idx + next_frame_skip), fontdict={'fontsize': 14})

            plt.show()


class Augmentation(object):
    def __init__(self, shift_limit=0.2, scale_limit=0.3, rotate_limit=20, apply_prob=1):
        self.aug = Compose([ShiftScaleRotate(
            shift_limit=shift_limit, scale_limit=scale_limit,
            rotate_limit=rotate_limit, border_mode=cv2.BORDER_CONSTANT, value=0)], p=apply_prob)

    def __call__(self, imgs):
        n = len(imgs)
        num = [0]
        for img in imgs:
            num.append(img.shape[2])
        data = {'image': np.concatenate(imgs, axis=2)}
        augmented = self.aug(**data)
        out_img = augmented['image']

        out = []
        indices = np.cumsum(num)
        for i in range(1, n+1):
            out.append(out_img[:, :, indices[i-1]: indices[i]])
        return out


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    # data_val = FullPicDataloader(transform=None, mode='val', viz=True, img_size=256,
    #                              num_skip=4, skip=2, max_num_events=60000)
    # print('number of samples', len(data_val))
    # for _ in range(1):
    #     i = np.random.randint(0, len(data_val))
    #     one_sample = data_val[i]
    #     for k, v in one_sample.items():
    #         print(k, v.shape)

    data = FlowDataloader(transform=Augmentation(), mode='train', viz=False, img_size=256, input_channel=1,
                          num_skip=4, skip=2, max_num_events=60000, source='events')

    # data = FlowDataloader(transform=None, mode='all', viz=False, img_size=256, input_channel=8,
    #                       num_skip=8, skip=1, max_num_events=60000, source='events')

    print('number of samples', len(data))
    i = np.random.randint(0, len(data))
    one_smple = data[i]
    print(one_smple['info'])
    for k, v in one_smple.items():
        if k != 'info':
            print(k, v.size())


