// src/services/aiService.js
import { Platform } from 'react-native';
import dataset from '../data/dataset.json';

const BASE_URL = Platform.OS === 'android' ? 'http://10.0.2.2:5000' : 'http://localhost:5000';

export const getSuggestion = async (questionText, answerText, context) => {
  try {
    const qMatch = dataset.questionnaire.find(q => q.question === questionText);
    if (qMatch) {
      const answerClean = answerText.split('(')[0].trim();
      const optMatch = qMatch.options.find(
        opt => opt.text.startsWith(answerClean) || answerClean.startsWith(opt.text)
      );
      if (optMatch && optMatch.suggestions && optMatch.suggestions.length > 0) {
        // Return a random suggestion from the clinical suggestions list
        const randomIndex = Math.floor(Math.random() * optMatch.suggestions.length);
        return optMatch.suggestions[randomIndex];
      }
    }
  } catch (error) {
    console.error("Error looking up suggestion locally:", error);
  }

  // Fallback suggestion
  return "ඔබගේ පිළිතුරට අනුව, ඔබ හොඳින් විවේක ගැනීම වැදගත් බව පෙනේ.";
};

export const getChatResponse = async (chatMessage, questionnaireContext, chatHistory) => {
  try {
    const formattedHistory = chatHistory.map(m => ({
      sender: m.sender,
      text: m.text
    }));
    
    // Add the current user message to history snapshot if it's not already in there
    const historySnapshot = [...formattedHistory, { sender: 'user', text: chatMessage }];

    // Format questionnaireContext to match backend's expected structure
    const formattedContext = questionnaireContext.map(q => ({
      questionId: q.questionId,
      question: q.questionText || q.question,
      answer: q.answer,
      score: q.score || 1,
      suggestions: q.suggestions || [q.suggestion]
    }));

    const response = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: chatMessage,
        contextHistory: formattedContext,
        chatHistory: historySnapshot,
        use_local_model: false // Default to Gemini on backend
      })
    });
    
    const data = await response.json();
    return data.reply || "දෝෂයකි.";
  } catch (error) {
    console.error("Chat response API error:", error);
    return "ජාල දෝෂයකි. කරුණාකර නැවත උත්සාහ කරන්න.";
  }
};

export const getFinalSummary = async (questionnaireContext, chatHistory) => {
  try {
    const formattedContext = questionnaireContext.map(q => ({
      questionId: q.questionId,
      question: q.questionText || q.question,
      answer: q.answer,
      score: q.score || 1,
      suggestions: q.suggestions || [q.suggestion]
    }));

    const formattedHistory = chatHistory.map(m => ({
      sender: m.sender,
      text: m.text
    }));

    const response = await fetch(`${BASE_URL}/api/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        answers: formattedContext,
        chatHistory: formattedHistory
      })
    });
    
    const data = await response.json();
    let filtered = data.suggestions || [];
    
    // Add urgent hotline if score was 5 anywhere
    const maxScore = Math.max(...formattedContext.map(c => c.score), 0);
    if (maxScore === 5 && !filtered.some(s => s.includes("1926"))) {
      filtered.unshift("🚨 කරුණාකර හැකි ඉක්මනින් මනෝවිද්‍යා උපදේශකයෙකුගේ සහාය ලබා ගන්න හෝ 1926 (ජාතික මානසික සෞඛ්‍ය උපකාරක සේවය) වෙත අමතන්න.");
    }

    // Format the suggestions as a numbered string list for displaying in standard Text component
    if (filtered.length > 0) {
      return filtered.map((s, idx) => `${idx + 1}. ${s}`).join('\n\n');
    }
    
    return "යෝජනා කිසිවක් හමු නොවීය.";
  } catch (error) {
    console.error("Final summary API error:", error);
    return "සම්පූර්ණ සංවාදයට අනුව මාගේ ප්‍රධාන යෝජනා:\n\n1. දිනපතා පැය 7-8 ක නින්දක් ලබා ගැනීමට උත්සාහ කරන්න.\n2. අධික පීඩනය හෝ කලබලකාරී බවක් දැනෙන සෑම අවස්ථාවකදීම ගැඹුරු හුස්ම ගැනීමේ ව්‍යායාම කරන්න.\n3. සිතට වද දෙන සිතුවිලි ඇත්නම් එය විශ්වාසවන්ත අයෙකු සමග බෙදා ගන්න.\n4. ඔබට තවදුරටත් මෙම අපහසුතා පවතී නම් වෘත්තීය මනෝවිද්‍යා උපදේශනයක් (Counseling) ලබා ගැනීමට පසුබට නොවන්න.";
  }
};
