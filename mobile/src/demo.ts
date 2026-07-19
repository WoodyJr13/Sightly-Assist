import type { AlertAction, MobileSystemState } from './types';

const actionMessage: Record<AlertAction, string> = {
  no_alert: 'Path clear',
  awareness_left: 'Hazard left',
  awareness_center: 'Hazard ahead',
  awareness_right: 'Hazard right',
  slow: 'Slow down',
  stop: 'Stop',
  abstain: 'Unable to assess',
};

export function createLocalDemoState(elapsedMs: number, frameId: number): MobileSystemState {
  const phase = (elapsedMs / 1000) % 14;
  let warningAction: AlertAction = 'no_alert';
  let warningReason = 'walking corridor is currently clear';
  let warningShouldEmit = false;
  let center: MobileSystemState['free_space']['center'] = 'clear';
  let hazards: MobileSystemState['hazards'] = [];

  if (phase >= 2 && phase < 9) {
    const progress = (phase - 2) / 7;
    const depth = Math.max(0.9, 5.5 - 4.6 * progress);
    const ttc = Math.max(0.45, depth / 1.35);
    const risk = Math.min(1, 0.18 + 0.82 * progress);

    if (risk >= 0.8 || ttc <= 1.5) {
      warningAction = 'stop';
      warningReason = 'approaching person intersects the center walking corridor';
      center = 'blocked';
    } else if (risk >= 0.6 || ttc <= 3) {
      warningAction = 'slow';
      warningReason = 'approaching person may enter the center walking corridor';
      center = 'constrained';
    } else {
      warningAction = 'awareness_center';
      warningReason = 'person detected ahead with increasing relative risk';
    }

    warningShouldEmit = Math.floor(elapsedMs / 250) % 6 === 0;
    hazards = [
      {
        track_id: 12,
        class_name: 'person',
        confidence: 0.94,
        bbox: {
          x_min: 0.39 - 0.05 * progress,
          y_min: 0.2 - 0.04 * progress,
          x_max: 0.61 + 0.05 * progress,
          y_max: 0.86 + 0.04 * progress,
        },
        depth_m: depth,
        velocity_x_mps: 0.03,
        velocity_z_mps: -1.35,
        speed_mps: 1.35,
        risk_score: risk,
        time_to_closest_approach_s: ttc,
        distance_at_closest_approach_m: 0.18,
        predicted_collision: risk >= 0.62,
        assessment_status: 'valid',
      },
    ];
  } else if (phase >= 9 && phase < 12) {
    const progress = (phase - 9) / 3;
    warningAction = 'awareness_left';
    warningReason = 'crossing person remains outside the center collision boundary';
    warningShouldEmit = Math.floor(elapsedMs / 500) % 4 === 0;
    hazards = [
      {
        track_id: 27,
        class_name: 'person',
        confidence: 0.91,
        bbox: {
          x_min: 0.05 + 0.22 * progress,
          y_min: 0.28,
          x_max: 0.27 + 0.22 * progress,
          y_max: 0.85,
        },
        depth_m: 2.7,
        velocity_x_mps: 0.95,
        velocity_z_mps: -0.05,
        speed_mps: 0.95,
        risk_score: 0.42,
        time_to_closest_approach_s: 1.1,
        distance_at_closest_approach_m: 1.25,
        predicted_collision: false,
        assessment_status: 'valid',
      },
    ];
  }

  return {
    schema_version: '1.0',
    source: 'phone-demo',
    sequence_id: 'expo-preview-loop',
    frame_id: frameId,
    timestamp_ns: Date.now() * 1_000_000,
    warning_action: warningAction,
    warning_should_emit: warningShouldEmit,
    warning_reason: warningReason,
    free_space: { left: 'clear', center, right: 'clear' },
    hazards,
    metrics: {
      processing_fps: 29.4,
      total_latency_ms: 34,
      detector_latency_ms: 17.8,
      tracker_latency_ms: 1.2,
      depth_latency_ms: 3.4,
      motion_latency_ms: 0.8,
      risk_latency_ms: 0.4,
    },
    synchronized: true,
    orientation_available: true,
    system_message: actionMessage[warningAction],
  };
}
