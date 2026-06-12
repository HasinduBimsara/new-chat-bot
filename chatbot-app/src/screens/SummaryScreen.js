import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator, Share } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { getFinalSummary } from '../services/aiService';

export default function SummaryScreen({ route, navigation }) {
  const { questionnaireContext, chatHistory, guestUid } = route.params;
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const result = await getFinalSummary(questionnaireContext, chatHistory);
        setSummary(result);
      } catch (error) {
        setSummary("සාරාංශය ලබාගැනීමේදී දෝෂයක් ඇතිවිය.");
      } finally {
        setLoading(false);
      }
    };
    
    fetchSummary();
  }, []);

  const handleShare = async () => {
    if (!summary) return;
    try {
      await Share.share({
        title: 'මානසික සෞඛ්‍ය උපදේශන වාර්තාව',
        message: `මානසික සෞඛ්‍ය උපදේශන වාර්තාව - අවසාන යෝජනා:\n\n${summary}`,
      });
    } catch (error) {
      console.error("Error sharing summary:", error);
    }
  };

  const handleRestart = () => {
    // Reset navigation stack to Welcome screen
    navigation.reset({
      index: 0,
      routes: [{ name: 'Welcome' }],
    });
  };

  return (
    <LinearGradient 
      colors={['#0f2027', '#203a43', '#2c5364']} 
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false}>
        <Text style={styles.title}>අවසාන යෝජනා (Top Suggestions)</Text>
        
        <View style={styles.glassCard}>
          {loading ? (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#00d2ff" />
              <Text style={styles.loadingText}>කෘතිම බුද්ධිය (AI) මගින් ඔබට වඩාත් සුදුසු යෝජනා තෝරමින් පවතී...</Text>
            </View>
          ) : (
            <Text style={styles.summaryText}>{summary}</Text>
          )}
        </View>

        {!loading && (
          <View style={styles.buttonContainer}>
            <TouchableOpacity 
              style={[styles.button, styles.shareButton]} 
              activeOpacity={0.8}
              onPress={handleShare}
            >
              <Text style={styles.buttonText}>යෝජනා Share කරන්න</Text>
            </TouchableOpacity>

            <TouchableOpacity 
              style={[styles.button, styles.restartButton]} 
              activeOpacity={0.8}
              onPress={handleRestart}
            >
              <Text style={styles.buttonText}>නැවත ආරම්භ කරන්න</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContainer: {
    flexGrow: 1,
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#00d2ff',
    marginBottom: 25,
    textAlign: 'center',
  },
  glassCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    padding: 25,
    borderRadius: 20,
    width: '100%',
    maxWidth: 450,
    marginBottom: 35,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.2,
    shadowRadius: 10,
    elevation: 5,
    minHeight: 200,
    justifyContent: 'center',
  },
  summaryText: {
    fontSize: 16,
    color: '#fff',
    lineHeight: 26,
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 10,
  },
  loadingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#b0c4de',
    textAlign: 'center',
    lineHeight: 24,
  },
  buttonContainer: {
    width: '100%',
    maxWidth: 450,
    gap: 15,
  },
  button: {
    paddingVertical: 15,
    borderRadius: 30,
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
    elevation: 3,
  },
  shareButton: {
    backgroundColor: '#2ecc71',
    shadowColor: '#2ecc71',
  },
  restartButton: {
    backgroundColor: '#3a7bd5',
    shadowColor: '#3a7bd5',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
