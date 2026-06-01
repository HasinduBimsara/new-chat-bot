import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';

export default function WelcomeScreen({ navigation }) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>ඔබේ මානසික සෞඛ්‍යය මිතුරා</Text>
      <Text style={styles.subtitle}>
        අපි ප්‍රශ්න කිහිපයකින් ආරම්භ කරමු. ඔබගේ පිළිතුරු ඉතා රහසිගතයි.
      </Text>
      
      <TouchableOpacity 
        style={styles.button} 
        onPress={() => navigation.navigate('Questionnaire')}
      >
        <Text style={styles.buttonText}>ආරම්භ කරන්න (Start)</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2c3e50',
    textAlign: 'center',
    marginBottom: 20,
  },
  subtitle: {
    fontSize: 16,
    color: '#34495e',
    textAlign: 'center',
    marginBottom: 40,
    lineHeight: 24,
  },
  button: {
    backgroundColor: '#3498db',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 25,
    elevation: 3,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
});
