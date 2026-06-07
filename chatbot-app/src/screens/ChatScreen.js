import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, FlatList, KeyboardAvoidingView, Platform, ActivityIndicator } from 'react-native';
import { getChatResponse } from '../services/aiService';
import { db, auth } from '../services/firebase';
import { collection, query, orderBy, onSnapshot, addDoc, serverTimestamp } from 'firebase/firestore';

export default function ChatScreen({ route, navigation }) {
  const { questionnaireContext } = route.params;
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const flatListRef = useRef(null);

  // Get current user UID or fallback to a default UID for development/testing
  const userUID = auth.currentUser?.uid || 'guest_user_123';
  const messagesCollectionRef = collection(db, 'users', userUID, 'messages');

  // Load chat history in real-time from Firestore
  useEffect(() => {
    const q = query(messagesCollectionRef, orderBy('timestamp', 'asc'));
    
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const loadedMessages = snapshot.docs.map(doc => {
        const data = doc.data();
        return {
          id: doc.id,
          text: data.text,
          sender: data.sender,
          timestamp: data.timestamp
        };
      });

      // If the database is empty (new user session), insert the default welcome message
      if (loadedMessages.length === 0) {
        addDoc(messagesCollectionRef, {
          text: 'ඔබට තවදුරටත් කතා කිරීමට අවශ්‍ය වෙනත් යමක් තිබේද? (ඔබට අවශ්‍ය නැතිනම් පහළ ඇති බොත්තම ඔබා අවසන් යෝජනා ලබා ගන්න)',
          sender: 'bot',
          timestamp: serverTimestamp()
        });
      } else {
        setMessages(loadedMessages);
      }
    }, (error) => {
      console.error("Firestore subscription error:", error);
    });

    return () => unsubscribe();
  }, [userUID]);

  // Scroll to bottom when messages change
  const scrollToBottom = () => {
    if (flatListRef.current && messages.length > 0) {
      flatListRef.current.scrollToEnd({ animated: true });
    }
  };

  const handleSend = async () => {
    if (!inputText.trim()) return;

    const typedText = inputText;
    setInputText('');
    setIsTyping(true);

    try {
      // 1. Save user's message to Firestore (reactive listener will render it immediately)
      await addDoc(messagesCollectionRef, {
        text: typedText,
        sender: 'user',
        timestamp: serverTimestamp()
      });

      // 2. Call backend with full chat history (which now includes this message)
      const currentHistory = [...messages, { text: typedText, sender: 'user' }];
      const responseText = await getChatResponse(typedText, questionnaireContext, currentHistory);

      // 3. Save bot's reply to Firestore
      await addDoc(messagesCollectionRef, {
        text: responseText,
        sender: 'bot',
        timestamp: serverTimestamp()
      });
    } catch (error) {
      console.error("Error sending message:", error);
      await addDoc(messagesCollectionRef, {
        text: 'කණගාටුයි, සන්නිවේදන දෝෂයක් ඇතිවිය.',
        sender: 'bot',
        timestamp: serverTimestamp()
      });
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
        ref={flatListRef}
        data={messages}
        keyExtractor={item => item.id}
        renderItem={renderMessage}
        contentContainerStyle={styles.chatList}
        onContentSizeChange={scrollToBottom}
        onLayout={scrollToBottom}
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
