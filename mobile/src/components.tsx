import React from 'react';
import { Pressable, StyleSheet, Text, View } from 'react-native';

import type {
  ConnectionStatus,
  CorridorStatus,
  MobileHazard,
} from './types';
import { corridorColor, formatNumber, palette } from './theme';

export function ConnectionPill({ status }: { status: ConnectionStatus }): React.JSX.Element {
  const color =
    status === 'connected' || status === 'demo'
      ? palette.green
      : status === 'connecting'
        ? palette.amber
        : palette.red;
  return (
    <View style={styles.connectionPill}>
      <View style={[styles.connectionDot, { backgroundColor: color }]} />
      <Text style={styles.connectionText}>{status.toUpperCase()}</Text>
    </View>
  );
}

export function SectionTitle({ title }: { title: string }): React.JSX.Element {
  return <Text style={styles.sectionTitle}>{title.toUpperCase()}</Text>;
}

export function CorridorCard({
  label,
  status,
  prominent = false,
}: {
  label: string;
  status: CorridorStatus;
  prominent?: boolean;
}): React.JSX.Element {
  const color = corridorColor(status);
  return (
    <View
      style={[
        styles.corridorCard,
        prominent && styles.corridorCardProminent,
        { borderColor: color },
      ]}
    >
      <Text style={styles.corridorLabel}>{label}</Text>
      <View style={[styles.corridorIndicator, { backgroundColor: color }]} />
      <Text style={[styles.corridorStatus, { color }]}>{status.toUpperCase()}</Text>
    </View>
  );
}

export function HazardCard({ hazard }: { hazard: MobileHazard }): React.JSX.Element {
  const riskColor = hazard.predicted_collision
    ? palette.red
    : (hazard.risk_score ?? 0) >= 0.6
      ? palette.amber
      : palette.cyan;
  return (
    <View style={styles.hazardCard}>
      <View style={styles.hazardHeader}>
        <View>
          <Text style={styles.hazardName}>{hazard.class_name.toUpperCase()}</Text>
          <Text style={styles.hazardId}>
            TRACK #{hazard.track_id} · {(hazard.confidence * 100).toFixed(0)}%
          </Text>
        </View>
        <View style={[styles.riskBadge, { borderColor: riskColor }]}>
          <Text style={[styles.riskBadgeText, { color: riskColor }]}>
            {hazard.risk_score === null
              ? 'UNSCORED'
              : `RISK ${(hazard.risk_score * 100).toFixed(0)}%`}
          </Text>
        </View>
      </View>
      <View style={styles.hazardMetrics}>
        <MiniMetric label="DISTANCE" value={formatNumber(hazard.depth_m, 'm')} />
        <MiniMetric label="SPEED" value={formatNumber(hazard.speed_mps, 'm/s')} />
        <MiniMetric
          label="CPA TIME"
          value={formatNumber(hazard.time_to_closest_approach_s, 's')}
        />
        <MiniMetric
          label="CPA GAP"
          value={formatNumber(hazard.distance_at_closest_approach_m, 'm')}
        />
      </View>
    </View>
  );
}

export function MetricCard({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <View style={styles.metricCard}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  );
}

export function EmptyCard({ text }: { text: string }): React.JSX.Element {
  return (
    <View style={styles.emptyCard}>
      <Text style={styles.mutedText}>{text}</Text>
    </View>
  );
}

export function SegmentButton({
  label,
  active,
  onPress,
}: {
  label: string;
  active: boolean;
  onPress: () => void;
}): React.JSX.Element {
  return (
    <Pressable
      style={[styles.segmentButton, active && styles.segmentButtonActive]}
      onPress={onPress}
    >
      <Text style={[styles.segmentText, active && styles.segmentTextActive]}>{label}</Text>
    </Pressable>
  );
}

export function TabButton({
  label,
  active,
  onPress,
}: {
  label: string;
  active: boolean;
  onPress: () => void;
}): React.JSX.Element {
  return (
    <Pressable style={styles.tabButton} onPress={onPress}>
      <View style={[styles.tabIndicator, active && styles.tabIndicatorActive]} />
      <Text style={[styles.tabLabel, active && styles.tabLabelActive]}>{label}</Text>
    </Pressable>
  );
}

function MiniMetric({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <View style={styles.miniMetric}>
      <Text style={styles.miniMetricLabel}>{label}</Text>
      <Text style={styles.miniMetricValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  connectionPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.border,
    paddingHorizontal: 10,
    paddingVertical: 7,
    borderRadius: 20,
  },
  connectionDot: { width: 8, height: 8, borderRadius: 4, marginRight: 7 },
  connectionText: { color: palette.text, fontSize: 10, fontWeight: '800' },
  sectionTitle: {
    color: palette.muted,
    fontSize: 11,
    fontWeight: '900',
    letterSpacing: 1.5,
    marginTop: 23,
    marginBottom: 10,
  },
  corridorCard: {
    flex: 1,
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderRadius: 14,
    paddingVertical: 13,
    paddingHorizontal: 8,
    alignItems: 'center',
  },
  corridorCardProminent: { backgroundColor: palette.surfaceRaised },
  corridorLabel: { color: palette.muted, fontSize: 9, fontWeight: '800' },
  corridorIndicator: { width: 9, height: 9, borderRadius: 5, marginTop: 9, marginBottom: 6 },
  corridorStatus: { fontSize: 10, fontWeight: '900' },
  hazardCard: {
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.border,
    borderRadius: 17,
    padding: 16,
    marginBottom: 10,
  },
  hazardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  hazardName: { color: palette.text, fontSize: 18, fontWeight: '900' },
  hazardId: { color: palette.muted, fontSize: 10, marginTop: 4 },
  riskBadge: { borderWidth: 1, borderRadius: 12, paddingHorizontal: 10, paddingVertical: 7 },
  riskBadgeText: { fontSize: 10, fontWeight: '900' },
  hazardMetrics: { flexDirection: 'row', flexWrap: 'wrap', marginTop: 15, gap: 8 },
  miniMetric: {
    width: '47%',
    flexGrow: 1,
    backgroundColor: '#0a1523',
    borderRadius: 11,
    padding: 10,
  },
  miniMetricLabel: { color: palette.muted, fontSize: 8, fontWeight: '800' },
  miniMetricValue: { color: palette.text, fontSize: 14, fontWeight: '700', marginTop: 4 },
  metricCard: {
    width: '48%',
    flexGrow: 1,
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.border,
    borderRadius: 15,
    padding: 15,
  },
  metricLabel: { color: palette.muted, fontSize: 9, fontWeight: '800' },
  metricValue: { color: palette.text, fontSize: 22, fontWeight: '800', marginTop: 7 },
  emptyCard: {
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.border,
    borderRadius: 15,
    padding: 18,
  },
  mutedText: { color: palette.muted, fontSize: 12 },
  segmentButton: { flex: 1, paddingVertical: 11, borderRadius: 10, alignItems: 'center' },
  segmentButtonActive: { backgroundColor: '#1a4960' },
  segmentText: { color: palette.muted, fontSize: 10, fontWeight: '900' },
  segmentTextActive: { color: palette.cyan },
  tabButton: { flex: 1, alignItems: 'center', paddingTop: 9, paddingBottom: 7 },
  tabIndicator: {
    width: 24,
    height: 3,
    borderRadius: 2,
    backgroundColor: 'transparent',
    marginBottom: 7,
  },
  tabIndicatorActive: { backgroundColor: palette.cyan },
  tabLabel: { color: palette.muted, fontSize: 11, fontWeight: '700' },
  tabLabelActive: { color: palette.text },
});
