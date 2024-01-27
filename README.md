# StarPy

I am using KinectFusion for Reconstruction and BundleTrack for Object Tracking.

## Installation

1. Compile BundleTrack. Install Dependency: `sudo apt install libpcl-dev`, `sudo apt-get install freeglut3-dev`

## Run tracking

### Generate 2D Tracking Mask

1. First we need to get the mask first using `pytracking`. Download `RTS` checkpoint at https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md#Models-1. Notice that pytracking requires `torch` version to be lower than `1.9` and `numpy` should be lower that `1.20`.

2. Run tracking. You can do the tracking using the following code:

```python
cd external/pytracking/pytracking
python run_video.py rts rts50 ../../../test_data/exp_5/color/%d.jpg --save_results
```

### Generate Hand Tracking

Download the mediapipe model first.
```bash
wget -q https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### Generate 6D pose tracking

```bash
./build/bundle_track_colmap config.yml
```

### Generate visualization

```bash
python external/BundleTrack/external/Easy3DViewer/example/python/check_sequence_poses.py --frame_skip=30 --traj_path=log/poses/ --data_path=test_data/exp_1/camera_2/
```

## Experiment Setup