import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { questions, options } from '../data/questions';
import dataset from '../data/dataset.json';
import { db, auth } from '../services/firebase';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';

export default function QuestionnaireScreen({ route, navigation }) {
  const guestUid = route.params?.guestUid;
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answersContext, setAnswersContext] = useState([]);

  const handleOptionSelect = async (option) => {
    const question = questions[currentQuestionIndex];
    
    // Find matching suggestions in dataset.json silently
    let suggestions = [];
    try {
      const qMatch = dataset.questionnaire.find(q => q.id === question.id);
      if (qMatch) {
        const answerClean = option.text.split('(')[0].trim();
        const optMatch = qMatch.options.find(
          opt => opt.text.startsWith(answerClean) || answerClean.startsWith(opt.text)
        );
        if (optMatch && optMatch.suggestions) {
          suggestions = optMatch.suggestions;
        }
      }
    } catch (e) {
      console.log("Error extracting suggestions silently:", e);
    }

    const newContext = [
      ...answersContext, 
      { 
        questionId: question.id, 
        question: question.text, 
        answer: option.text,
        score: option.id, // option.id represents the score (1 to 5)
        suggestions: suggestions
      }
    ];
    setAnswersContext(newContext);

    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      // Save questionnaire to Firestore
      try {
        const userUID = auth.currentUser?.uid || guestUid || 'guest_user_123';
        const userDocRef = doc(db, 'users', userUID);
        await setDoc(userDocRef, {
          questionnaireAnswers: newContext,
          lastUpdated: serverTimestamp()
        }, { merge: true });
        console.log("Mobile app questionnaire answers saved to Firestore.");
      } catch (error) {
        console.error("Error saving mobile questionnaire to Firestore:", error);
      }

      // Go to Chat phase
      navigation.navigate('Chat', { questionnaireContext: newContext, guestUid });
    }
  };

  const currentQuestion = questions[currentQuestionIndex];

  return (
    <LinearGradient 
      colors={['#0f2027', '#203a43', '#2c5364']} 
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContainer} showsVerticalScrollIndicator={false}>
        <Text style={styles.progressText}>ප්‍රශ්නය {currentQuestionIndex + 1} / {questions.length}</Text>
        
        <View style={styles.glassCard}>
          <Text style={styles.questionText}>{currentQuestion.text}</Text>
        </View>

        <View style={styles.optionsContainer}>
          {options.map((option) => (
            <TouchableOpacity 
              key={option.id} 
              style={styles.optionButton}
              activeOpacity={0.7}
              onPress={() => handleOptionSelect(option)}
            >
              <Text style={styles.optionText}>{option.text}</Text>
            </TouchableOpacity>
          ))}
        </View>
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
  progressText: {
    fontSize: 14,
    color: '#b0c4de',
    marginBottom: 15,
    fontWeight: 'bold',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  glassCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    padding: 25,
    borderRadius: 20,
    width: '100%',
    maxWidth: 450,
    marginBottom: 25,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.2,
    shadowRadius: 10,
    elevation: 5,
  },
  questionText: {
    fontSize: 18,
    color: '#fff',
    lineHeight: 28,
    textAlign: 'center',
  },
  optionsContainer: {
    width: '100%',
    maxWidth: 450,
    gap: 12,
  },
  optionButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    paddingVertical: 16,
    paddingHorizontal: 20,
    borderRadius: 14,
    width: '100%',
  },
  optionText: {
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
    fontWeight: '500',
  },
});
