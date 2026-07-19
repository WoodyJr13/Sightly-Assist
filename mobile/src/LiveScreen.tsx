import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { EmptyCard, HazardCard, SectionTitle } from './components';
import { palette, warningStyle } from './theme';
import type { MobileHazard, MobileSystemState } from './types';

export function LiveScreen({ state }: { state: MobileSystemState }): React.JSX.Element {
  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View style={styles.previewHeader}>
        <View>
          <Text style={styles.heading}>LIVE PERCEPTION</Text>
          <Text style={styles.muted}>{state.sequence_id}</Text>
        </View>
        <Text style={styles.frame}>FRAME {state.frame_id}</Text>
      </View>

      <View style={styles.cameraPreview}>
        <View style={[styles.guideLine, styles.guideVertical]} />
        <View style={[styles.guideLine, styles.guideHorizontal]} />
        <View style={styles.walkingCorridor} />
        {state.hazards.map((hazard) => (
          <HazardOverlay key={hazard.track_id} hazard={hazard} />
        ))}
        <View style={styles.previewBadge}>
          <Text style={styles.previewBadgeText}>{state.source.toUpperCase()}</Text>
        </View>
      </View>

      <View style={styles.warningStrip}>
        <Text
          style={[
            styles.warningText,
            { color: warningStyle[state.warning_action].color },
          ]}
        >
          {warningStyle[state.warning_action].label}
        </Text>
        <Text style={styles.warningReason}>{state.warning_reason}</Text>
      </View>

      <SectionTitle title={`Tracked objects (${state.hazards.length})`} />
      {state.hazards.length === 0 ? (
        <EmptyCard text="No active tracks." />
      ) : (
        state.hazards.map((hazard) => <HazardCard key={hazard.track_id} hazard={hazard} />)
      )}

      <SectionTitle title="Stage latency" />
      <View style={styles.latencyCard}>
        <LatencyBar label="Detector" value={state.metrics.detector_latency_ms} max={40} />
        <LatencyBar label="Tracker" value={state.metrics.tracker_latency_ms} max={40} />
        <LatencyBar label="Depth" value={state.metrics.depth_latency_ms} max={40} />
        <LatencyBar label="Motion" value={state.metrics.motion_latency_ms} max={40} />
        <LatencyBar label="Risk" value={state.metrics.risk_latency_ms} max={40} />
      </View>
    </ScrollView>
  );
}

function HazardOverlay({ hazard }: { hazard: MobileHazard }): React.JSX.Element {
  const color = hazard.predicted_collision
    ? palette.red
    : (hazard.risk_score ?? 0) >= 0.6
      ? palette.amber
      : palette.cyan;
  const left = `${hazard.bbox.x_min * 100}%` as `${number}%`;
  const top = `${hazard.bbox.y_min * 100}%` as `${number}%`;
  const width = `${(hazard.bbox.x_max - hazard.bbox.x_min) * 100}%` as `${number}%`;
  const height = `${(hazard.bbox.y_max - hazard.bbox.y_min) * 100}%` as `${number}%`;
  return (
    <View style={[styles.overlayBox, { left, top, width, height, borderColor: color }]}>
      <View style={[styles.overlayLabel, { backgroundColor: color }]}>
        <Text style={styles.overlayLabelText}>
          #{hazard.track_id} {hazard.class_name} {hazard.depth_m?.toFixed(1) ?? '?'}m
        </Text>
      </View>
    </View>
  );
}

function LatencyBar({
  label,
  value,
  max,
}: {
  label: string;
  value: number;
  max: number;
}): React.JSX.Element {
  const width = `${Math.min(100, (value / max) * 100)}%` as `${number}%`;
  return (
    <View style={styles.latencyRow}>
      <View style={styles.latencyLabels}>
        <Text style={styles.latencyLabel}>{label}</Text>
        <Text style={styles.latencyValue}>{value.toFixed(1)} ms</Text>
      </View>
      <View style={styles.latencyTrack}>
        <View style={[styles.latencyFill, { width }]} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  content: { padding: 18, paddingBottom: 36 },
  previewHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
    marginBottom: 12,
  },
  heading: { color: palette.text, fontSize: 17, fontWeight: '900', letterSpacing: 1 },
  muted: { color: palette.muted, fontSize: 12, marginTop: 3 },
  frame: { color: palette.cyan, fontSize: 10, fontWeight: '900' },
  cameraPreview: {
    aspectRatio: 4 / 3,
    backgroundColor: '#03070d',
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#31506d',
    overflow: 'hidden',
    position: 'relative',
  },
  guideLine: { position: 'absolute', backgroundColor: 'rgba(77,217,255,0.16)' },
  guideVertical: { width: 1, height: '100%', left: '50%' },
  guideHorizontal: { height: 1, width: '100%', top: '50%' },
  walkingCorridor: {
    position: 'absolute',
    left: '34%',
    right: '34%',
    top: '45%',
    bottom: 0,
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: 'rgba(62,227,143,0.45)',
    backgroundColor: 'rgba(62,227,143,0.05)',
  },
  overlayBox: { position: 'absolute', borderWidth: 2, borderRadius: 4 },
  overlayLabel: {
    position: 'absolute',
    left: -2,
    top: -22,
    paddingHorizontal: 5,
    paddingVertical: 3,
    borderRadius: 3,
  },
  overlayLabelText: { color: '#07101d', fontSize: 9, fontWeight: '900' },
  previewBadge: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(7,16,29,0.8)',
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 5,
  },
  previewBadgeText: { color: palette.cyan, fontSize: 8, fontWeight: '900' },
  warningStrip: {
    backgroundColor: palette.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: palette.border,
    marginTop: 12,
    padding: 14,
  },
  warningText: { fontSize: 16, fontWeight: '900' },
  warningReason: { color: palette.muted, fontSize: 11, lineHeight: 16, marginTop: 4 },
  latencyCard: {
    backgroundColor: palette.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: palette.border,
    padding: 15,
  },
  latencyRow: { marginBottom: 13 },
  latencyLabels: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  latencyLabel: { color: palette.text, fontSize: 12 },
  latencyValue: { color: palette.muted, fontSize: 11 },
  latencyTrack: { height: 7, borderRadius: 4, backgroundColor: '#1a2c41', overflow: 'hidden' },
  latencyFill: { height: '100%', borderRadius: 4, backgroundColor: palette.cyan },
});
