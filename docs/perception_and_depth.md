# Perception and Depth Pipeline

## Scope

This milestone adds a hardware-independent recorded-data path from decoded RGB and depth frames to tracked objects with camera-frame 3D positions.

```text
RGB frame
  -> ONNX detector
  -> class-aware NMS
  -> persistent tracker
  -> synchronized depth sampling
  -> robust object distance
  -> camera-frame x, y, z
```

The output is not yet a collision decision. Motion estimation and ego-motion compensation must be added before 3D observations are connected to the existing closest-approach risk model.

## ONNX detector

`OnnxYoloDetector` uses ONNX Runtime and accepts a lightweight YOLO-style detection model.

Supported model output layouts:

- raw Ultralytics output: `N x (4 + class_count)` containing center-x, center-y, width, height, and class scores;
- raw YOLOv5-style output: `N x (5 + class_count)` containing box values, objectness, and class scores;
- end-to-end NMS output: `N x 6` containing x-min, y-min, x-max, y-max, confidence, and class ID.

The backend performs:

- aspect-ratio-preserving letterbox resize;
- BGR-to-RGB conversion;
- NCHW float32 normalization;
- ONNX Runtime inference;
- confidence filtering;
- configurable class filtering;
- class-aware non-maximum suppression;
- restoration of boxes to source-image pixels.

The default class list is COCO. The default hazard filter includes people, road vehicles, bicycles, dogs, benches, bags, sports balls, skateboards, chairs, couches, plants, and tables. These defaults are provisional and must be evaluated against the project dataset.

Install runtime dependencies with:

```bash
python -m pip install -e ".[detector]"
```

Model weights are intentionally not committed to Git. A compatible ONNX file must be placed under the ignored model directory or supplied from another local path. This prevents large binary files and model licensing from being hidden in source-control history.

## Depth association

Each tracked bounding box is reduced to a configurable central region. The central region is used because box edges frequently contain background pixels.

The association process:

1. clips the region to image boundaries;
2. converts sensor values using the configured depth scale;
3. removes NaN, infinite, too-near, and too-far samples;
4. requires a minimum number of valid samples;
5. estimates an initial median and median absolute deviation;
6. rejects large depth outliers using a robust MAD band;
7. reports median depth, valid fraction, inlier count, and residual MAD;
8. deprojects the bounding-box center using calibrated pinhole intrinsics.

Camera coordinates use:

- positive `x`: image right;
- positive `y`: image down;
- positive `z`: forward from the camera.

For pixel `(u, v)` and depth `z`:

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
```

Depth maps are currently loaded as numeric two-dimensional NumPy arrays. Their height and width must match the RGB frame and camera calibration.

## Validation

Automated tests cover:

- real ONNX Runtime session loading and inference using a generated detector graph;
- raw and end-to-end detector output layouts;
- coordinate restoration after letterboxing;
- confidence and class filtering;
- class-aware NMS;
- missing or malformed RGB/depth data;
- robust median depth under invalid pixels and extreme outliers;
- insufficient-depth status;
- camera deprojection;
- replay integration from decoded images through tracking and depth association.

## Remaining limitations

- The repository does not yet include a benchmarked production weight file.
- The baseline tracker is IoU-based; ByteTrack or another motion-aware tracker still needs comparison.
- Depth is associated per frame without temporal filtering.
- Camera motion is not compensated.
- 3D velocity, acceleration, covariance, and trajectory uncertainty are not yet estimated.
- No real-world mobility or safety claim is supported by this milestone.
