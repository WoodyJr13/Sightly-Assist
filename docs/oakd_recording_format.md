# OAK-D Recording Format

## Purpose

This format creates reproducible RGB-D and IMU sequences that can be processed by the existing replay pipeline without requiring the camera to remain connected. The recorder is intended for controlled engineering experiments, not live mobility decisions.

## Supported runtime

The OAK adapter targets DepthAI v3 and pins `depthai==3.7.1`. It uses the current unified `Camera` node, `StereoDepth`, `ImageAlign`, `Sync`, and `IMU` APIs.

Primary implementation references:

- Luxonis DepthAI v3 documentation
- Luxonis Camera node documentation
- Luxonis StereoDepth documentation
- Luxonis Sync documentation
- Luxonis IMU documentation

No DepthAI or stereo-matching implementation is claimed as original work.

## Directory layout

```text
<sequence>/
├── manifest.json
├── capture_metadata.json
├── rgb/
│   ├── frame_00000000.png
│   └── ...
├── depth/
│   ├── frame_00000000.npy
│   └── ...
└── imu/
    └── samples.jsonl
```

### RGB

RGB frames are stored as lossless PNG files in BGR channel order when loaded through OpenCV. Lossless storage avoids introducing JPEG artifacts into later detector comparisons.

### Depth

Depth frames are stored as two-dimensional NumPy arrays with `float32` values in meters. Invalid stereo pixels remain zero unless the device runtime produces another invalid representation. Downstream depth association rejects zero, nonfinite, too-near, and too-far values.

### IMU

`samples.jsonl` contains one validated JSON object per IMU sample. A sample may include:

- acceleration in meters per second squared;
- angular velocity in radians per second;
- fused orientation in `w, x, y, z` order;
- orientation accuracy;
- device sequence number;
- timestamp in nanoseconds.

The recorder requests uncalibrated accelerometer and gyroscope reports because DepthAI expresses those processed reports in the Luxonis right/down/forward frame. Fused orientation is requested only when configured and available on the connected IMU.

### Manifest

`manifest.json` stores:

- sequence identifier;
- source identifier;
- format version;
- camera intrinsics;
- frame IDs and timestamps;
- RGB and depth paths;
- synchronization status and measured timestamp difference;
- device sequence number when available;
- frame orientation when available;
- SHA-256 checksums for every RGB and depth file;
- IMU and metadata paths.

## Atomic recording behavior

The recorder writes into a temporary sibling directory. The destination is created only after all frames, metadata, checksums, and the manifest have been written successfully. An exception removes the temporary directory. Existing output is rejected unless `--overwrite` is explicit.

## Command

```text
sightly record-oakd \
  --sequence-id hallway-head-on-001 \
  --output datasets/hallway-head-on-001 \
  --duration 10 \
  --width 640 \
  --height 400 \
  --fps 30
```

Use `--no-orientation` when testing a device whose IMU does not provide a fused rotation vector. Use `--max-frames` when a fixed frame count is more important than duration.

## Required experiment notes

The raw capture directory does not by itself describe the experimental condition. Each formal dataset must also record:

- scenario identifier from the experiment matrix;
- date and location;
- lighting and surface conditions;
- camera mounting position and orientation;
- measured course geometry;
- object or mannequin trajectory;
- whether any people appear in the recording;
- eligibility, consent, and privacy determination;
- operator and reviewer;
- device firmware and calibration status;
- any dropped frames, USB warnings, or thermal issues.

## Known limitations

- Device hardware is not available in GitHub Actions, so CI validates the pinned DepthAI import, packet parsing, dataset writer, and replay contract but cannot validate sensor output.
- Fused orientation availability depends on the IMU fitted to the device.
- RGB/depth alignment and camera intrinsics must be checked on the specific device before experiments.
- Host and device timestamp behavior must be measured on the final Jetson/OAK configuration.
- PNG and NumPy storage prioritize research fidelity over disk efficiency.
- The current recorder does not yet capture exposure, gain, focus, temperature, USB speed, dropped-packet counters, or power measurements.
