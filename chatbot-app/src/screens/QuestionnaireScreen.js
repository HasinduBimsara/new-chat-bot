import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, ActivityIndicator } from 'react-native';
import { questions, options } from '../data/questions';
import { getSuggestion } from '../services/aiService';

export default function QuestionnaireScreen({ navigation }) {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answersContext, setAnswersContext] = useState([]);
  const [loadingSuggestion, setLoadingSuggestion] = useState(false);
  const [currentSuggestion, setCurrentSuggestion] = useState(null);

  const handleOptionSelect = async (option) => {
    setLoadingSuggestion(true);
    setCurrentSuggestion(null);

    const question = questions[currentQuestionIndex];
    
    try {
      // Fetch suggestion from AI service (Mock SLM integration)
      const suggestion = await getSuggestion(question.text, option.text, answersContext);
      
      setCurrentSuggestion(suggestion);
      
      const newContext = [
        ...answersContext, 
        { 
          questionId: question.id, 
          questionText: question.text, 
          answer: option.text,
          suggestion: suggestion
        }
      ];
      setAnswersContext(newContext);
    } catch (error) {
      console.error(error);
      setCurrentSuggestion("තාක්ෂණික දෝෂයකි. කරුණාකර පසුව උත්සාහ කරන්න.");
    } finally {
      setLoadingSuggestion(false);
    }
  };

  const handleNext = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
      setCurrentSuggestion(null);
    } else {
      // Go to Chat phase
      navigation.navigate('Chat', { questionnaireContext: answersContext });
    }
  };

  const currentQuestion = questions[currentQuestionIndex];

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.progressText}>ප්‍රශ්නය {currentQuestionIndex + 1} / 10</Text>
      
      <View style={styles.questionCard}>
        <Text style={styles.questionText}>{currentQuestion.text}</Text>
      </View>

      {!currentSuggestion && !loadingSuggestion && (
        <View style={styles.optionsContainer}>
          {options.map((option) => (
            <TouchableOpacity 
              key={option.id} 
              style={styles.optionButton}
              onPress={() => handleOptionSelect(option)}
            >
              <Text style={styles.optionText}>{option.text}</Text>
            </TouchableOpacity>
          ))}
        </View>
      )}

      {loadingSuggestion && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#3498db" />
          <Text style={styles.loadingText}>යෝජනාව සකස් කරමින්...</Text>
        </View>
      )}

      {currentSuggestion && !loadingSuggestion && (
        <View style={styles.suggestionContainer}>
          <Text style={styles.suggestionTitle}>AI යෝජනාව:</Text>
          <Text style={styles.suggestionText}>{currentSuggestion}</Text>
          
          <TouchableOpacity style={styles.nextButton} onPress={handleNext}>
            <Text style={styles.nextButtonText}>
              {currentQuestionIndex < questions.length - 1 ? "මීළඟ ප්‍රශ්නය" : "සාකච්ඡාවට යන්න (Chat)"}
            </Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    backgroundColor: '#f5f7fa',
    padding: 20,
    alignItems: 'center',
  },
  progressText: {
    fontSize: 16,
    color: '#7f8c8d',
    marginBottom: 20,
    fontWeight: 'bold',
  },
  questionCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 15,
    width: '100%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    marginBottom: 30,
  },
  questionText: {
    fontSize: 18,
    color: '#2c3e50',
    lineHeight: 28,
    textAlign: 'center',
  },
  optionsContainer: {
    width: '100%',
  },
  optionButton: {
    backgroundColor: '#ecf0f1',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#bdc3c7',
  },
  optionText: {
    fontSize: 16,
    color: '#34495e',
    textAlign: 'center',
  },
  loadingContainer: {
    marginTop: 40,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#7f8c8d',
    fontSize: 16,
  },
  suggestionContainer: {
    backgroundColor: '#e8f8f5',
    padding: 20,
    borderRadius: 15,
    width: '100%',
    borderWidth: 1,
    borderColor: '#1abc9c',
  },
  suggestionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#16a085',
    marginBottom: 10,
  },
  suggestionText: {
    fontSize: 16,
    color: '#2c3e50',
    lineHeight: 24,
    marginBottom: 20,
  },
  nextButton: {
    backgroundColor: '#1abc9c',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  nextButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
