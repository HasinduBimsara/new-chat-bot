import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TextInput, TouchableOpacity, FlatList, KeyboardAvoidingView, Platform, ActivityIndicator, Switch } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { getChatResponse } from '../services/aiService';
import { db, auth } from '../services/firebase';
import { collection, query, orderBy, onSnapshot, addDoc, serverTimestamp } from 'firebase/firestore';

export default function ChatScreen({ route, navigation }) {
  const { questionnaireContext, guestUid } = route.params;
  const DEFAULT_WELCOME_TEXT = 'ප්‍රශ්නාවලිය අවසන්! ඔබට තවදුරටත් මා සමග කතා කිරීමට අවශ්‍ය වෙනත් යමක් තිබේද? (අවශ්‍ය නැතිනම් පහළ ඇති බොත්තම ඔබා අවසන් යෝජනා ලබා ගන්න)';
  
  const [messages, setMessages] = useState([
    {
      id: 'welcome',
      text: DEFAULT_WELCOME_TEXT,
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [useLocalModel, setUseLocalModel] = useState(false); // Default to Gemini AI (false)
  const flatListRef = useRef(null);

  // Get current user UID or fallback to the passed guest session UID
  const userUID = auth.currentUser?.uid || guestUid || 'guest_user_123';
  const messagesCollectionRef = collection(db, 'users', userUID, 'messages');

  // Load chat history in real-time from Firestore, fallback to local state if permissions fail
  useEffect(() => {
    let unsubscribe = () => {};
    try {
      const q = query(messagesCollectionRef, orderBy('timestamp', 'asc'));
      
      unsubscribe = onSnapshot(q, (snapshot) => {
        const loadedMessages = snapshot.docs.map(doc => {
          const data = doc.data();
          return {
            id: doc.id,
            text: data.text,
            sender: data.sender,
            timestamp: data.timestamp
          };
        });

        if (loadedMessages.length > 0) {
          setMessages(loadedMessages);
        } else {
          // Try to seed the welcome message to Firestore in the background
          addDoc(messagesCollectionRef, {
            text: DEFAULT_WELCOME_TEXT,
            sender: 'bot',
            timestamp: serverTimestamp()
          }).catch(err => console.log("Silent Firestore seed error (using local welcome message):", err));
        }
      }, (error) => {
        console.warn("Firestore subscription error (falling back to local chat):", error.message);
      });
    } catch (err) {
      console.warn("Firestore initialization error (falling back to local chat):", err.message);
    }

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

    // Create user message for local state immediately
    const tempUserMsgId = "user_" + Math.random().toString(36).substring(2, 11);
    const userMsg = {
      id: tempUserMsgId,
      text: typedText,
      sender: 'user',
      timestamp: new Date()
    };
    
    // Update local state immediately so user sees their message
    setMessages(prev => {
      // Prevent duplicates in case Firestore updates quickly
      if (prev.some(m => m.text === typedText && m.sender === 'user')) return prev;
      return [...prev, userMsg];
    });

    try {
      // 1. Try to save user's message to Firestore in background
      addDoc(messagesCollectionRef, {
        text: typedText,
        sender: 'user',
        timestamp: serverTimestamp()
      }).catch(err => console.log("Silent Firestore write error:", err));

      // 2. Call backend with full chat history (including this message) and local model flag
      const currentHistory = [...messages, { text: typedText, sender: 'user' }];
      const responseText = await getChatResponse(typedText, questionnaireContext, currentHistory, useLocalModel);

      // Create bot message for local state
      const tempBotMsgId = "bot_" + Math.random().toString(36).substring(2, 11);
      const botMsg = {
        id: tempBotMsgId,
        text: responseText,
        sender: 'bot',
        timestamp: new Date()
      };

      // Update local state so user sees bot response
      setMessages(prev => {
        // Prevent duplicates in case Firestore updates quickly
        if (prev.some(m => m.text === responseText && m.sender === 'bot')) return prev;
        return [...prev, botMsg];
      });

      // 3. Try to save bot's reply to Firestore in background
      addDoc(messagesCollectionRef, {
        text: responseText,
        sender: 'bot',
        timestamp: serverTimestamp()
      }).catch(err => console.log("Silent Firestore write error:", err));

    } catch (error) {
      console.error("Error sending message:", error);
      
      const errorMsg = {
        id: "err_" + Math.random().toString(36).substring(2, 11),
        text: 'ජාල දෝෂයකි. කරුණාකර නැවත උත්සාහ කරන්න.',
        sender: 'bot',
        timestamp: new Date()
      };
      
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
    <LinearGradient 
      colors={['#0f2027', '#203a43', '#2c5364']} 
      style={styles.container}
    >
      <KeyboardAvoidingView 
        style={styles.flexContainer} 
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        {/* Model Selector Toggle */}
        <View style={styles.toggleContainer}>
          <Text style={[styles.toggleLabel, !useLocalModel && styles.activeToggleLabel]}>Gemini AI</Text>
          <Switch
            value={useLocalModel}
            onValueChange={setUseLocalModel}
            trackColor={{ false: '#334155', true: '#38bdf8' }}
            thumbColor={useLocalModel ? '#fff' : '#f4f3f4'}
          />
          <Text style={[styles.toggleLabel, useLocalModel && styles.activeToggleLabel]}>Local Model (30M)</Text>
        </View>

        <FlatList
          ref={flatListRef}
          data={messages}
          keyExtractor={item => item.id}
          renderItem={renderMessage}
          contentContainerStyle={styles.chatList}
          onContentSizeChange={scrollToBottom}
          onLayout={scrollToBottom}
          showsVerticalScrollIndicator={false}
        />
        
        {isTyping && (
          <View style={styles.typingIndicator}>
            <ActivityIndicator size="small" color="#00d2ff" />
            <Text style={styles.typingText}>Bot පිළිතුරක් සකස් කරමින් පවතී...</Text>
          </View>
        )}

        <View style={styles.inputOuterContainer}>
          <View style={styles.inputContainer}>
            <TextInput
              style={styles.input}
              placeholder="ඔබේ අදහස මෙහි ලියන්න..."
              placeholderTextColor="#b0c4de"
              value={inputText}
              onChangeText={setInputText}
              multiline
            />
            <TouchableOpacity style={styles.sendButton} onPress={handleSend} activeOpacity={0.8}>
              <Text style={styles.sendButtonText}>➤</Text>
            </TouchableOpacity>
          </View>

          <TouchableOpacity 
            style={styles.finishButton} 
            activeOpacity={0.8}
            onPress={() => navigation.navigate('Summary', { questionnaireContext, chatHistory: messages, guestUid })}
          >
            <Text style={styles.finishButtonText}>අවසන් කර යෝජනා ලබා ගන්න</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  flexContainer: {
    flex: 1,
  },
  toggleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    gap: 12,
  },
  toggleLabel: {
    fontSize: 14,
    color: '#b0c4de',
  },
  activeToggleLabel: {
    color: '#00d2ff',
    fontWeight: 'bold',
  },
  chatList: {
    padding: 15,
    paddingBottom: 25,
  },
  messageBubble: {
    maxWidth: '80%',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 18,
    marginBottom: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#3a7bd5',
    borderBottomRightRadius: 4,
  },
  botBubble: {
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(255, 255, 255, 0.08)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    borderBottomLeftRadius: 4,
  },
  messageText: {
    fontSize: 16,
    color: '#fff',
    lineHeight: 22,
  },
  typingIndicator: {
    flexDirection: 'row',
    paddingHorizontal: 15,
    paddingVertical: 8,
    alignItems: 'center',
    backgroundColor: 'rgba(0,0,0,0.1)',
    gap: 8,
  },
  typingText: {
    color: '#00d2ff',
    fontSize: 14,
  },
  inputOuterContainer: {
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    paddingTop: 10,
    paddingBottom: Platform.OS === 'ios' ? 25 : 10,
    paddingHorizontal: 10,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.1)',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 10,
  },
  input: {
    flex: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 22,
    paddingHorizontal: 18,
    paddingVertical: 10,
    fontSize: 16,
    color: '#fff',
    maxHeight: 100,
  },
  sendButton: {
    backgroundColor: '#00d2ff',
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#00d2ff',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
    elevation: 3,
  },
  sendButtonText: {
    color: '#0f2027',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: -2, // Visual alignment of the arrow icon
  },
  finishButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.15)',
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
  },
  finishButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
