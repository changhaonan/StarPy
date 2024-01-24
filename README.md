# StarPy

I am using KinectFusion for Reconstruction and BundleTrack for Object Tracking.

## Installation

1. Compile BundleTrack. Install Dependency: `sudo apt install libpcl-dev`, `sudo apt-get install freeglut3-dev`

## Run tracking

### Generate 2D Tracking Mask

1. First we need to get the mask first using `pytracking`. Download `RTS` checkpoint at https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md#Models-1. Notice that pytracking requires `torch` version to be lower than `1.9` and `numpy` should be lower that `1.20`.

2. Run tracking. You can do the tracking using the following code:

```python
python run_video.py rts rts50 ../../../test_data/exp_0/camera_2/color/%d.jpg --save_results
```

### Generate 6D pose tracking

```bash
./build/bundle_track_colmap config.yml
```

## Experiment Setup