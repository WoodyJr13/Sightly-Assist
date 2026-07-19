# Research Question and Hypotheses

## Project title

**Sightly Assist: Uncertainty-Aware Edge AI for Predictive Pedestrian Hazard Detection**

## Davidson classification

- Domain: Artificial Intelligence
- Specific field: Edge AI for assistive computer vision and predictive collision-risk estimation

## Primary research question

Can a wearable, locally running RGB-D perception system predict near-term pedestrian collision hazards with greater warning lead time and fewer unnecessary alerts than detection-only and distance-threshold baselines?

## Secondary research questions

1. How much do persistent tracking and temporally filtered 3D motion improve collision-event precision and time-to-closest-approach accuracy?
2. How much does camera-motion compensation reduce false motion and false warnings while the wearer translates or rotates?
3. Does uncertainty-aware abstention reduce confident unsafe errors without making the system unusably silent?
4. Does depth-derived free-space context reduce false directional awareness alerts in cluttered scenes?
5. Which detector, tracker, and model-runtime combination provides the best accuracy-latency-power tradeoff on edge hardware?

## Primary hypothesis

A full system combining object detection, persistent tracking, robust depth association, temporally filtered relative 3D motion, camera-motion compensation, closest-point-of-approach geometry, uncertainty gating, and a constrained warning policy will outperform simpler baselines on collision-event precision, warning lead time, and false alerts per minute.

## Supporting hypotheses

- Temporal filtering will reduce stationary-object velocity error compared with frame-to-frame differencing.
- Camera-motion compensation will reduce false approach estimates caused by wearer motion.
- Uncertainty gating will reduce high-confidence false warnings and missed-risk claims, at the cost of a measurable abstention rate.
- Free-space context will improve interpretation of whether a detected object actually obstructs the likely walking corridor.
- A lightweight edge model will meet a useful real-time latency target without cloud inference.

## Original contribution boundary

The project does not claim novelty in object detection, optical flow, depth cameras, or ONNX inference individually. The proposed contribution is the safety-oriented system that converts noisy RGB-D observations into conservative, uncertainty-aware predictions of future path conflict on edge hardware.

## Safety boundary

The system is an experimental research prototype. It may issue only awareness, slow, stop, or abstain outputs. It must not provide autonomous steering instructions or replace a cane, guide dog, trained mobility technique, or certified assistive device.
