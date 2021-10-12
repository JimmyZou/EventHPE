import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib


def prepare_full_pic(data_dir, target_dir, image_size=512):
    # ----------- collect actions ----------------
    action_names = []
    # filter out group4_time1
    for action in sorted(os.listdir(data_dir)):
        if 'group4' not in action and 'subject' in action:
            action_names.append(action)
    print('-----------------------------------------')
    print('prepare data for the following action_names:\n    ' + '\n    '.join(action_names))
    print('-----------------------------------------')

    for action in action_names:
        if not os.path.exists('%s/full_pic_%i/%s' % (target_dir, image_size, action)):
            os.mkdir('%s/full_pic_%i/%s' % (target_dir, image_size, action))

        fnames = sorted(glob.glob('%s/%s/event_camera/full_pic/*.jpg' % (data_dir, action)))
        for fname in fnames:
            img = cv2.imread(fname)[:, :, 0]  # [1280, 800]
            img = np.pad(img, ((0, 0), (240, 240)), 'constant')  # [1280, 1280]

            # print(img.shape)
            # plt.figure(figsize=(12, 8))
            # plt.imshow(img, cmap='gray')
            # plt.axis('off')
            # plt.show()

            img = cv2.resize(img, (image_size, image_size))
            img_name = fname.split('/')[-1]
            cv2.imwrite('%s/full_pic_%i/%s/%s' % (target_dir, image_size, action, img_name), img)
        print('finish %s' % action)


def generate_events_frame(events, num_partitions=4, h=1280, w=800):
    # events [N, 3], v, u, in_pixel_time
    start_time = events[0, 2]
    end_time = events[-1, 2]
    time_window = (end_time - start_time) / num_partitions
    events_frames = []
    for i in range(num_partitions):
        events_frame = np.zeros([h, w])
        idx = (events[:, 2] >= (start_time + i * time_window)) & (events[:, 2] < (start_time + (i + 1) * time_window))
        v = h - 1 - events[idx, 0]
        u = w - 1 - events[idx, 1]
        events_frame[v, u] = 1
        events_frames.append(events_frame)
    events_frames = np.stack(events_frames, axis=2)
    return events_frames


def prepare_event_frame_single(actions, cpu_id, data_dir, target_dir, num_partitions=4, h=1280, w=800, image_size=256):
    for action in actions:
        if not os.path.exists('%s/events_%i/%s' % (target_dir, image_size, action)):
            os.mkdir('%s/events_%i/%s' % (target_dir, image_size, action))

        print('[cpu %i] finish %s' % (cpu_id, action))
        frame_indices, events_count = [], []
        fnames = sorted(glob.glob('%s/%s/event_camera/events/event*.csv' % (data_dir, action)))[:-1]
        # count = 0
        for fname in fnames:
            # count += 1
            # if count < 1065:
            #     continue

            events = pd.read_csv(fname, header=None, dtype=np.int32,
                                 names=['v', 'u', 'in_pixel_time', 'off_pixel_time', 'polarity'])
            events = events[['v', 'u', 'in_pixel_time']].values
            img = generate_events_frame(events, num_partitions, h, w)
            # print(np.sum((img > 0) & (img < 1)))
            # print(events.shape[0])
            # print(img.shape, np.max(img), np.min(img))

            # plt.figure(figsize=(16, 8))
            # for i in range(4):
            #     plt.subplot(1, 4, i+1)
            #     plt.imshow(img[:, :, i], cmap='gray')
            #     plt.title('# events: %i' % np.sum(img[:, :, i]), fontdict={'fontsize': 12})
            #     # plt.axis('off')
            # plt.show()

            img = np.pad(img, ((0, 0), (240, 240), (0, 0)), 'constant')  # [1280, 1280, 4]
            img_resize = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

            # print(img_resize.shape)
            # print(np.max(img_resize), np.min(img_resize), np.sum((img_resize>0) & (img_resize<1)))
            # plt.figure(figsize=(16, 8))
            # for i in range(4):
            #     plt.subplot(1, 4, i+1)
            #     plt.imshow(img_resize[:, :, i], cmap='gray')
            #     plt.title('# events: %i' % np.sum(img_resize[:, :, i] > 0), fontdict={'fontsize': 12})
            #     # plt.axis('off')
            # plt.show()

            img_name = fname.split('/')[-1].split('.')[0]
            cv2.imwrite('%s/events_%i/%s/%s.png' % (target_dir, image_size, action, img_name),
                        img_resize.astype(np.uint8))

            # tmp = cv2.imread('%s/events/%s/%s.png' % (target_dir, action, img_name), -1)
            # plt.figure(figsize=(16, 8))
            # for i in range(4):
            #     plt.subplot(1, 4, i + 1)
            #     plt.imshow(tmp[:, :, i], cmap='gray')
            #     plt.title('# events: %i' % np.sum(tmp[:, :, i] > 0), fontdict={'fontsize': 12})
            #     # plt.axis('off')
            # plt.show()

            # print(np.sum(img > 0), events.shape[0])
            frame_idx = int(fname.split('/')[-1].split('.')[0].replace('event', ''))
            frame_indices.append(frame_idx)
            events_count.append(np.sum(img_resize > 0))
            # break
        # break

        joblib.dump([frame_indices, events_count],
                    '%s/events_%i/%s/%s_info.pkl' % (target_dir, image_size, action, action), compress=3)


def prepare_event_frame(data_dir, target_dir, num_partitions=4, h=1280, w=800, image_size=256, num_cpus=4):
    # ----------- collect actions ----------------
    action_names = []
    # filter out group4_time1
    for action in sorted(os.listdir(data_dir)):
        if 'group4' not in action and 'subject' in action:
            action_names.append(action)
    print('-----------------------------------------')
    print('prepare data for the following action_names:\n    ' + '\n    '.join(action_names))
    print('-----------------------------------------')

    # prepare_event_frame_single(['subject02_group2_time1'])

    import multiprocessing
    N = len(action_names)
    n_files_cpu = N // num_cpus + 1

    results = []
    pool = multiprocessing.Pool(num_cpus)
    for i in range(num_cpus):
        idx1 = i * n_files_cpu
        idx2 = min((i + 1) * n_files_cpu, N)
        results.append(pool.apply_async(prepare_event_frame_single,
                                        (action_names[idx1:idx2], i, data_dir, target_dir,
                                         num_partitions, h, w, image_size)))
    pool.close()
    pool.join()
    pool.terminate()

    for result in results:
        tmp = result.get()
        if tmp is not None:
            print(tmp)
    print('Multi-cpu pre-processing ends.')


def count_events(target_dir, new):
    all_files = []
    action_names = sorted(os.listdir('%s/full_pic' % target_dir))
    for action in action_names:
        # if 'subject01_group1_time1' not in action:
        #     continue
        print(action)
        if not os.path.exists('%s/events/%s/%s_info.pkl' % (target_dir, action, action)) or new:
            fnames = list(sorted(glob.glob('%s/full_pic/%s/*.jpg' % (target_dir, action))))[:-1]
            if len(fnames) == 0:
                continue
            indices, events_count = [], []
            for fname in fnames:
                frame_idx = int(fname.split('/')[-1].split('.')[0].replace('fullpic', ''))
                event_frame = cv2.imread('%s/events/%s/event%04i.png' % (target_dir, action, frame_idx), -1)
                num_events = np.sum(event_frame > 0)
                indices.append(frame_idx)
                events_count.append(num_events)
                if event_frame is None:
                    print('[warning] %s/events/%s/event%04i.png' % (target_dir, action, frame_idx))
            joblib.dump([indices, events_count], '%s/events/%s/%s_info.pkl' % (target_dir, action, action), compress=3)


def prepare_color(data_dir, target_dir, image_size):
    # ----------- collect actions ----------------
    action_names = []
    # filter out group4_time1
    for action in sorted(os.listdir(data_dir)):
        if 'group4' not in action and 'subject' in action:
            action_names.append(action)
    print('-----------------------------------------')
    print('prepare data for the following action_names:\n    ' + '\n    '.join(action_names))
    print('-----------------------------------------')

    if not os.path.exists('%s/color_%i' % (target_dir, image_size)):
        os.mkdir('%s/color_%i' % (target_dir, image_size))

    for action in action_names:
        if not os.path.exists('%s/color_%i/%s' % (target_dir, image_size, action)):
            os.mkdir('%s/color_%i/%s' % (target_dir, image_size, action))

        fnames = sorted(glob.glob('%s/%s/azure_kinect_0/color/*.jpg' % (data_dir, action)))
        for fname in fnames:
            img = cv2.imread(fname)[150:1300, 400:1550]  # [1150, 1150]

            # plt.figure(figsize=(12, 8))
            # plt.imshow(img)
            # plt.axis('off')
            # plt.show()

            img = cv2.resize(img, (image_size, image_size))
            img_name = fname.split('/')[-1]
            cv2.imwrite('%s/color_%i/%s/%s' % (target_dir, image_size, action, img_name), img)
        print('finish %s' % action)


if __name__ == '__main__':
    data_dir = '/data/shihao/data_event'
    target_dir = '/home/shihao/data_event/'
    # prepare_full_pic(data_dir, target_dir, image_size=512)
    # prepare_event_frame(data_dir, target_dir, image_size=512, num_cpus=4)
    # count_events(target_dir, new=True)
    # prepare_color(data_dir, target_dir, image_size=512)
    from flow_net.flowlib import viz_flow_colormap
    viz_flow_colormap(256)



