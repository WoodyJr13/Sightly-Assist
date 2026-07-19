import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import React from 'react';
import {
  Pressable,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  View,
} from 'react-native';

import { SectionTitle, SegmentButton } from './components';
import { palette } from './theme';
import type { ConnectionMode, ConnectionStatus } from './types';

export interface SettingsScreenProps {
  mode: ConnectionMode;
  status: ConnectionStatus;
  apiUrl: string;
  lastError: string | null;
  speechEnabled: boolean;
  hapticsEnabled: boolean;
  onApiUrlChange: (value: string) => void;
  onModeChange: (mode: ConnectionMode) => void;
  onReconnect: () => void;
  onSpeechChange: (enabled: boolean) => void;
  onHapticsChange: (enabled: boolean) => void;
}

export function SettingsScreen(props: SettingsScreenProps): React.JSX.Element {
  const testWarning = () => {
    if (props.speechEnabled) {
      Speech.speak('Stop', { language: 'en-US', rate: 0.88 });
    }
    if (props.hapticsEnabled) {
      void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
      <SectionTitle title="Data source" />
      <View style={styles.segmentedControl}>
        <SegmentButton
          label="PHONE DEMO"
          active={props.mode === 'local-demo'}
          onPress={() => props.onModeChange('local-demo')}
        />
        <SegmentButton
          label="BACKEND"
          active={props.mode === 'backend'}
          onPress={() => props.onModeChange('backend')}
        />
      </View>
      <Text style={styles.helpText}>
        Phone Demo runs without a computer. Backend mode connects over the same Wi-Fi network.
      </Text>

      <SectionTitle title="Backend address" />
      <View style={styles.card}>
        <Text style={styles.inputLabel}>COMPUTER LAN URL</Text>
        <TextInput
          value={props.apiUrl}
          onChangeText={props.onApiUrlChange}
          autoCapitalize="none"
          autoCorrect={false}
          keyboardType="url"
          placeholder="http://192.168.1.100:8765"
          placeholderTextColor="#60758c"
          style={styles.textInput}
        />
        <Text style={styles.helpText}>Use the computer IPv4 address, not localhost.</Text>
        <Pressable
          style={styles.primaryButton}
          onPress={() => {
            props.onModeChange('backend');
            props.onReconnect();
          }}
        >
          <Text style={styles.primaryButtonText}>CONNECT / RETRY</Text>
        </Pressable>
        <Text style={styles.connectionDetail}>Current status: {props.status}</Text>
        {props.lastError ? <Text style={styles.errorText}>{props.lastError}</Text> : null}
      </View>

      <SectionTitle title="Warning output" />
      <View style={styles.card}>
        <ToggleRow
          label="Spoken warnings"
          value={props.speechEnabled}
          onChange={props.onSpeechChange}
        />
        <ToggleRow
          label="Haptic warnings"
          value={props.hapticsEnabled}
          onChange={props.onHapticsChange}
        />
        <Pressable style={styles.secondaryButton} onPress={testWarning}>
          <Text style={styles.secondaryButtonText}>TEST STOP WARNING</Text>
        </Pressable>
      </View>

      <SectionTitle title="Safety status" />
      <View style={styles.noticeCard}>
        <Text style={styles.noticeTitle}>RESEARCH PROTOTYPE</Text>
        <Text style={styles.noticeText}>
          Use this preview only in controlled demonstrations. The system has not completed real-world safety validation.
        </Text>
      </View>
    </ScrollView>
  );
}

function ToggleRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
}): React.JSX.Element {
  return (
    <View style={styles.toggleRow}>
      <Text style={styles.toggleLabel}>{label}</Text>
      <Switch
        value={value}
        onValueChange={onChange}
        trackColor={{ false: '#30445c', true: '#1f7183' }}
        thumbColor={value ? palette.cyan : '#91a0af'}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  content: { padding: 18, paddingBottom: 36 },
  segmentedControl: {
    flexDirection: 'row',
    backgroundColor: palette.surface,
    borderRadius: 14,
    padding: 4,
    borderWidth: 1,
    borderColor: palette.border,
  },
  card: {
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.border,
    borderRadius: 17,
    padding: 16,
  },
  inputLabel: {
    color: palette.muted,
    fontSize: 9,
    fontWeight: '900',
    letterSpacing: 1,
    marginBottom: 8,
  },
  textInput: {
    backgroundColor: '#081321',
    color: palette.text,
    borderWidth: 1,
    borderColor: '#304a65',
    borderRadius: 12,
    paddingHorizontal: 13,
    paddingVertical: 12,
    fontSize: 14,
  },
  helpText: { color: palette.muted, fontSize: 11, lineHeight: 17, marginTop: 9 },
  primaryButton: {
    backgroundColor: palette.cyan,
    borderRadius: 12,
    alignItems: 'center',
    paddingVertical: 13,
    marginTop: 14,
  },
  primaryButtonText: { color: '#07101d', fontSize: 11, fontWeight: '900' },
  secondaryButton: {
    borderWidth: 1,
    borderColor: palette.red,
    borderRadius: 12,
    alignItems: 'center',
    paddingVertical: 12,
    marginTop: 14,
  },
  secondaryButtonText: { color: palette.red, fontSize: 11, fontWeight: '900' },
  connectionDetail: { color: palette.muted, fontSize: 10, marginTop: 12 },
  errorText: { color: palette.red, fontSize: 11, lineHeight: 16, marginTop: 7 },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  toggleLabel: { color: palette.text, fontSize: 14 },
  noticeCard: {
    backgroundColor: '#2b2113',
    borderColor: palette.amber,
    borderWidth: 1,
    borderRadius: 16,
    padding: 16,
  },
  noticeTitle: { color: palette.amber, fontSize: 12, fontWeight: '900', letterSpacing: 1 },
  noticeText: { color: '#e6d5b7', fontSize: 12, lineHeight: 18, marginTop: 7 },
});
