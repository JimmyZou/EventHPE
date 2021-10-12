## EventHPE: Event-based 3D Human Pose and Shape Estimation

Shihao Zou, Chuan Guo, Xinxin Zuo, Sen Wang, Xiaoqin Hu, Shoushun Chen, Minglun Gong and Li Cheng. ICCV 2021.


### Dataset
You can download the data from [Google Drive](https://drive.google.com/drive/folders/11gMj-5sgSiBciWNR0V6r9PMpru84zMk5?usp=sharing) 
or [Microsoft OneDrive](), 
which consists of
- preprocessed data
  - events_256 (event frames converted from raw events data, resolution 256x256)
  - full_pic_256 (gray-scale images)
  - pose_events (annotated poses of gray-scale images)
  - hmr_results (inferred poses of gray-scale images using [HMR](https://github.com/akanazawa/hmr))
  - vibe_results_0802 (inferred poses of gray-scale images using [VIBE](https://github.com/mkocabas/VIBE))
  - pred_flow_events_256 (inferred optical flow from event frames)
  - model (train/test on a snippet of 8 frames)
- raw events data

### Requirements
```
python 3.7.5
torch 1.7.0
opendr 0.78 (for render SMPL shape, installed successfully only under ubuntu 18.04)
cv2 4.1.1
```

To download the SMPL model go to [this](https://smpl.is.tue.mpg.de/) project website and 
register to get access to the downloads section. Place under __./smpl_model__. The model 
version used in our project is
```
basicModel_m_lbs_10_207_0_v1.0.0.pkl
basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
```

### Citation
If you would like to use our code or dataset, please cite either
```
@inproceedings{zou2021eventhpe,  
  title={EventHPE: Event-based 3D Human Pose and Shape Estimation},  
  author={Zou, Shihao and Guo, Chuan and Zuo, Xinxin and Wang, Sen and Xiaoqin, Hu and Chen, Shoushun and Gong, Minglun and Cheng, Li},  
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},  
  year={2021}  
} 
```

