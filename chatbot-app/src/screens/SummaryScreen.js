import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, ActivityIndicator } from 'react-native';
import { getFinalSummary } from '../services/aiService';

export default function SummaryScreen({ route, navigation }) {
  const { questionnaireContext, chatHistory } = route.params;
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

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>ඔබේ අවසාන යෝජනා (Top Suggestions)</Text>
      
      <View style={styles.summaryCard}>
        {loading ? (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#2ecc71" />
            <Text style={styles.loadingText}>යෝජනා සකස් කරමින්...</Text>
          </View>
        ) : (
          <Text style={styles.summaryText}>{summary}</Text>
        )}
      </View>

      <TouchableOpacity 
        style={styles.homeButton} 
        onPress={() => navigation.navigate('Welcome')}
      >
        <Text style={styles.homeButtonText}>මුල් පිටුවට යන්න</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#f5f7fa',
    padding: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2c3e50',
    marginBottom: 30,
    textAlign: 'center',
  },
  summaryCard: {
    backgroundColor: '#fff',
    padding: 25,
    borderRadius: 15,
    width: '100%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 40,
    minHeight: 200,
    justifyContent: 'center',
  },
  summaryText: {
    fontSize: 18,
    color: '#2c3e50',
    lineHeight: 28,
  },
  loadingContainer: {
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 15,
    fontSize: 16,
    color: '#7f8c8d',
  },
  homeButton: {
    backgroundColor: '#3498db',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 25,
  },
  homeButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
