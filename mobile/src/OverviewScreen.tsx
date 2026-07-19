import React, { useMemo } from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import {
  CorridorCard,
  EmptyCard,
  HazardCard,
  MetricCard,
  SectionTitle,
} from './components';
import { palette, warningStyle } from './theme';
import type { MobileSystemState } from './types';

export function OverviewScreen({
  state,
  history,
}: {
  state: MobileSystemState;
  history: MobileSystemState[];
}): React.JSX.Element {
  const warning = warningStyle[state.warning_action];
  const highestRisk = useMemo(
    () =>
      [...state.hazards].sort(
        (left, right) => (right.risk_score ?? -1) - (left.risk_score ?? -1),
      )[0],
    [state.hazards],
  );
  const recent = history.slice(-5).reverse();

  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <View
        style={[
          styles.warningCard,
          { backgroundColor: warning.background, borderColor: warning.color },
        ]}
      >
        <Text style={[styles.warningLabel, { color: warning.color }]}>{warning.label}</Text>
        <Text style={styles.warningMessage}>{state.system_message}</Text>
        <Text style={styles.warningReason}>{state.warning_reason}</Text>
      </View>

      <SectionTitle title="Scene status" />
      <View style={styles.corridorRow}>
        <CorridorCard label="LEFT" status={state.free_space.left} />
        <CorridorCard label="CENTER" status={state.free_space.center} prominent />
        <CorridorCard label="RIGHT" status={state.free_space.right} />
      </View>

      <SectionTitle title="Highest-priority object" />
      {highestRisk ? (
        <HazardCard hazard={highestRisk} />
      ) : (
        <EmptyCard text="No tracked hazards in the current frame." />
      )}

      <SectionTitle title="Performance" />
      <View style={styles.metricGrid}>
        <MetricCard label="PIPELINE" value={`${state.metrics.processing_fps.toFixed(1)} FPS`} />
        <MetricCard label="END TO END" value={`${state.metrics.total_latency_ms.toFixed(1)} ms`} />
        <MetricCard label="DETECTOR" value={`${state.metrics.detector_latency_ms.toFixed(1)} ms`} />
        <MetricCard label="RISK" value={`${state.metrics.risk_latency_ms.toFixed(1)} ms`} />
      </View>

      <SectionTitle title="System integrity" />
      <View style={styles.integrityCard}>
        <IntegrityRow label="RGB-depth synchronized" good={state.synchronized} />
        <IntegrityRow label="Orientation available" good={state.orientation_available} />
        <IntegrityRow label="Telemetry schema" good={state.schema_version === '1.0'} />
        <Text style={styles.sourceText}>
          Source: {state.source} · Frame {state.frame_id}
        </Text>
      </View>

      <SectionTitle title="Recent states" />
      <View style={styles.timelineCard}>
        {recent.length === 0 ? (
          <Text style={styles.muted}>Waiting for telemetry.</Text>
        ) : (
          recent.map((item) => (
            <View key={`${item.sequence_id}-${item.frame_id}`} style={styles.timelineRow}>
              <View
                style={[
                  styles.timelineDot,
                  { backgroundColor: warningStyle[item.warning_action].color },
                ]}
              />
              <Text style={styles.timelineAction}>
                {warningStyle[item.warning_action].label}
              </Text>
              <Text style={styles.timelineFrame}>#{item.frame_id}</Text>
            </View>
          ))
        )}
      </View>
    </ScrollView>
  );
}

function IntegrityRow({ label, good }: { label: string; good: boolean }): React.JSX.Element {
  return (
    <View style={styles.integrityRow}>
      <Text style={styles.integrityLabel}>{label}</Text>
      <Text style={[styles.integrityValue, { color: good ? palette.green : palette.amber }]}>
        {good ? 'READY' : 'UNAVAILABLE'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  content: { padding: 18, paddingBottom: 36 },
  warningCard: {
    borderWidth: 1.5,
    borderRadius: 22,
    padding: 22,
    minHeight: 168,
    justifyContent: 'center',
  },
  warningLabel: { fontSize: 34, fontWeight: '900', letterSpacing: 1.4 },
  warningMessage: { color: palette.text, fontSize: 21, fontWeight: '700', marginTop: 10 },
  warningReason: { color: '#d1dbe6', fontSize: 13, lineHeight: 19, marginTop: 7 },
  corridorRow: { flexDirection: 'row', gap: 8 },
  metricGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10 },
  integrityCard: {
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.border,
    borderRadius: 16,
    padding: 15,
  },
  integrityRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8 },
  integrityLabel: { color: palette.text, fontSize: 13 },
  integrityValue: { fontSize: 10, fontWeight: '900', letterSpacing: 0.7 },
  sourceText: {
    color: palette.muted,
    fontSize: 10,
    marginTop: 9,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: palette.border,
  },
  timelineCard: {
    backgroundColor: palette.surface,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: palette.border,
    padding: 13,
  },
  timelineRow: { flexDirection: 'row', alignItems: 'center', paddingVertical: 7 },
  timelineDot: { width: 8, height: 8, borderRadius: 4, marginRight: 10 },
  timelineAction: { color: palette.text, flex: 1, fontSize: 12, fontWeight: '700' },
  timelineFrame: { color: palette.muted, fontSize: 10 },
  muted: { color: palette.muted, fontSize: 12 },
});
