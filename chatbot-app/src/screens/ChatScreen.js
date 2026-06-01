import React, { useState } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, FlatList, KeyboardAvoidingView, Platform, ActivityIndicator } from 'react-native';
import { getChatResponse } from '../services/aiService';

export default function ChatScreen({ route, navigation }) {
  const { questionnaireContext } = route.params;
  const [messages, setMessages] = useState([
    { id: '1', text: 'ඔබට තවදුරටත් කතා කිරීමට අවශ්‍ය වෙනත් යමක් තිබේද? (ඔබට අවශ්‍ය නැතිනම් පහළ ඇති බොත්තම ඔබා අවසන් යෝජනා ලබා ගන්න)', sender: 'bot' }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async () => {
    if (!inputText.trim()) return;

    const userMessage = { id: Date.now().toString(), text: inputText, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    try {
      const responseText = await getChatResponse(userMessage.text, questionnaireContext, messages);
      const botMessage = { id: (Date.now() + 1).toString(), text: responseText, sender: 'bot' };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      const errorMsg = { id: (Date.now() + 1).toString(), text: 'කණගාටුයි, දෝෂයක් ඇතිවිය.', sender: 'bot' };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const renderMessage = ({ item }) => (
    <View style={[styles.messageBubble, item.sender === 'user' ? styles.userBubble : styles.botBubble]}>
      <Text style={styles.messageText}>{item.text}</Text>
    </View>
  );

  return (
    <KeyboardAvoidingView 
      style={styles.container} 
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={90}
    >
      <FlatList
        data={messages}
        keyExtractor={item => item.id}
        renderItem={renderMessage}
        contentContainerStyle={styles.chatList}
      />
      
      {isTyping && (
        <View style={styles.typingIndicator}>
          <ActivityIndicator size="small" color="#3498db" />
          <Text style={styles.typingText}>Bot Type කරමින් පවතී...</Text>
        </View>
      )}

      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          placeholder="ඔබේ අදහස මෙහි ලියන්න..."
          value={inputText}
          onChangeText={setInputText}
          multiline
        />
        <TouchableOpacity style={styles.sendButton} onPress={handleSend}>
          <Text style={styles.sendButtonText}>Send</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity 
        style={styles.finishButton} 
        onPress={() => navigation.navigate('Summary', { questionnaireContext, chatHistory: messages })}
      >
        <Text style={styles.finishButtonText}>සාකච්ඡාව අවසන් කර සම්පූර්ණ යෝජනා ලබා ගන්න</Text>
      </TouchableOpacity>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f7fa',
  },
  chatList: {
    padding: 15,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 15,
    marginBottom: 10,
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#3498db',
    borderBottomRightRadius: 0,
  },
  botBubble: {
    alignSelf: 'flex-start',
    backgroundColor: '#fff',
    borderBottomLeftRadius: 0,
    borderWidth: 1,
    borderColor: '#bdc3c7',
  },
  messageText: {
    fontSize: 16,
    color: '#2c3e50',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 10,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#ecf0f1',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    backgroundColor: '#ecf0f1',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    fontSize: 16,
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: '#3498db',
    padding: 12,
    borderRadius: 20,
    marginLeft: 10,
  },
  sendButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  typingIndicator: {
    flexDirection: 'row',
    padding: 10,
    alignItems: 'center',
  },
  typingText: {
    marginLeft: 10,
    color: '#7f8c8d',
  },
  finishButton: {
    backgroundColor: '#2ecc71',
    padding: 15,
    alignItems: 'center',
  },
  finishButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
