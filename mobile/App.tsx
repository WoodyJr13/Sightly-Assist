import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech';
import React, { useEffect, useRef, useState } from 'react';
import { SafeAreaView, StyleSheet, Text, View } from 'react-native';

import { ConnectionPill, TabButton } from './src/components';
import { LiveScreen } from './src/LiveScreen';
import { OverviewScreen } from './src/OverviewScreen';
import { SettingsScreen } from './src/SettingsScreen';
import { palette } from './src/theme';
import { useSightlyConnection } from './src/useSightlyConnection';

type TabName = 'overview' | 'live' | 'settings';

export default function App(): React.JSX.Element {
  const connection = useSightlyConnection();
  const [activeTab, setActiveTab] = useState<TabName>('overview');
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [hapticsEnabled, setHapticsEnabled] = useState(true);
  const lastCue = useRef<string | null>(null);

  useEffect(() => {
    const state = connection.state;
    if (!state.warning_should_emit) return;
    const cueId = `${state.sequence_id}:${state.frame_id}:${state.warning_action}`;
    if (lastCue.current === cueId) return;
    lastCue.current = cueId;

    if (speechEnabled) {
      void Speech.stop().then(() => {
        Speech.speak(state.system_message, {
          language: 'en-US',
          rate: state.warning_action === 'stop' ? 0.88 : 0.96,
          pitch: 1,
        });
      });
    }

    if (hapticsEnabled) {
      const feedback =
        state.warning_action === 'stop'
          ? Haptics.NotificationFeedbackType.Error
          : state.warning_action === 'slow'
            ? Haptics.NotificationFeedbackType.Warning
            : Haptics.NotificationFeedbackType.Success;
      void Haptics.notificationAsync(feedback);
    }
  }, [connection.state, hapticsEnabled, speechEnabled]);

  return (
    <SafeAreaView style={styles.safeArea}>
      <View style={styles.header}>
        <View>
          <Text style={styles.brand}>SIGHTLY ASSIST</Text>
          <Text style={styles.subtitle}>Predictive hazard monitor</Text>
        </View>
        <ConnectionPill status={connection.status} />
      </View>

      <View style={styles.content}>
        {activeTab === 'overview' ? (
          <OverviewScreen state={connection.state} history={connection.history} />
        ) : null}
        {activeTab === 'live' ? <LiveScreen state={connection.state} /> : null}
        {activeTab === 'settings' ? (
          <SettingsScreen
            mode={connection.mode}
            status={connection.status}
            apiUrl={connection.apiUrl}
            lastError={connection.lastError}
            speechEnabled={speechEnabled}
            hapticsEnabled={hapticsEnabled}
            onApiUrlChange={connection.setApiUrl}
            onModeChange={connection.setMode}
            onReconnect={connection.reconnect}
            onSpeechChange={setSpeechEnabled}
            onHapticsChange={setHapticsEnabled}
          />
        ) : null}
      </View>

      <View style={styles.tabBar}>
        <TabButton
          label="Overview"
          active={activeTab === 'overview'}
          onPress={() => setActiveTab('overview')}
        />
        <TabButton
          label="Live"
          active={activeTab === 'live'}
          onPress={() => setActiveTab('live')}
        />
        <TabButton
          label="Settings"
          active={activeTab === 'settings'}
          onPress={() => setActiveTab('settings')}
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: palette.background },
  header: {
    paddingHorizontal: 20,
    paddingTop: 12,
    paddingBottom: 14,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    borderBottomWidth: 1,
    borderBottomColor: palette.border,
  },
  brand: { color: palette.text, fontSize: 20, fontWeight: '900', letterSpacing: 1.5 },
  subtitle: { color: palette.muted, fontSize: 12, marginTop: 3 },
  content: { flex: 1 },
  tabBar: {
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: palette.border,
    backgroundColor: '#091422',
    paddingBottom: 8,
  },
});
