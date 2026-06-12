import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

export default function WelcomeScreen({ navigation }) {
  const [userUid] = useState(() => {
    return "app_guest_" + Math.random().toString(36).substring(2, 11);
  });

  return (
    <LinearGradient 
      colors={['#0f2027', '#203a43', '#2c5364']} 
      style={styles.container}
    >
      <View style={styles.glassCard}>
        <Text style={styles.title}>ඔබේ මානසික සෞඛ්‍යය මිතුරා</Text>
        <Text style={styles.subtitle}>
          අපි ප්‍රශ්න කිහිපයකින් ආරම්භ කරමු. මෙහි ලබාදෙන පිළිතුරු ඉතා රහසිගත වන අතර, ඔබගේ පිළිතුරු සියල්ල සලකා බලා අවසානයේදී ඔබට ගැලපෙනම යෝජනා ලබා දෙනු ඇත.
        </Text>
        
        <TouchableOpacity 
          style={styles.button} 
          activeOpacity={0.8}
          onPress={() => navigation.navigate('Questionnaire', { guestUid: userUid })}
        >
          <Text style={styles.buttonText}>ආරම්භ කරන්න</Text>
        </TouchableOpacity>
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  glassCard: {
    width: '100%',
    maxWidth: 400,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 24,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    padding: 30,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    elevation: 8,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#00d2ff',
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 36,
  },
  subtitle: {
    fontSize: 16,
    color: '#b0c4de',
    textAlign: 'center',
    marginBottom: 35,
    lineHeight: 26,
  },
  button: {
    backgroundColor: '#3a7bd5',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 30,
    shadowColor: '#00d2ff',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 10,
    elevation: 5,
    width: '100%',
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 0.5,
  },
});
