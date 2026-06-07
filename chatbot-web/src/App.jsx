import React, { useState, useRef, useEffect } from 'react';
import html2pdf from 'html2pdf.js';
import dataset from './dataset.json';
import { db, auth } from './firebase';
import { collection, query, orderBy, onSnapshot, addDoc, serverTimestamp, doc, setDoc } from 'firebase/firestore';

function App() {
  const [phase, setPhase] = useState('welcome'); // welcome, questionnaire, chat, summary
  const [currentQIndex, setCurrentQIndex] = useState(0);
  
  // History of answers
  const [contextHistory, setContextHistory] = useState([]);
  
  // Persistent local UID for testing, falls back to auth.currentUser?.uid if logged in
  const [userUid] = useState(() => {
    const cached = localStorage.getItem("chat_user_uid");
    if (cached) return cached;
    const generated = "web_guest_" + Math.random().toString(36).substr(2, 9);
    localStorage.setItem("chat_user_uid", generated);
    return generated;
  });

  // Chat state (Firestore messages will populate this)
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [useLocalModel, setUseLocalModel] = useState(false); // Default to Gemini
  const messagesEndRef = useRef(null);

  // Set up references to Firestore message collection
  const activeUid = auth.currentUser?.uid || userUid;
  const messagesCollectionRef = collection(db, 'users', activeUid, 'messages');

  // Real-time listener for Firestore chat messages
  useEffect(() => {
    if (phase !== 'chat') return;

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

      // Insert default welcome message if the database is empty
      if (loadedMessages.length === 0) {
        addDoc(messagesCollectionRef, {
          text: "ප්‍රශ්නාවලිය අවසන්! ඔබට තවදුරටත් මා සමග කතා කිරීමට අවශ්‍ය වෙනත් යමක් තිබේද? (අවශ්‍ය නැතිනම් 'අවසන් කරන්න' බොත්තම ඔබන්න)",
          sender: "bot",
          timestamp: serverTimestamp()
        });
      } else {
        setMessages(loadedMessages);
      }
    }, (error) => {
      console.error("Firestore loading error:", error);
    });

    return () => unsubscribe();
  }, [phase, activeUid]);

  // Scroll to bottom in chat
  useEffect(() => {
    if (phase === 'chat') {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, phase]);

  const handleStart = () => {
    setPhase('questionnaire');
  };

  const [finalSuggestions, setFinalSuggestions] = useState([]);
  const [isLoadingSummary, setIsLoadingSummary] = useState(false);

  const saveQuestionnaireToFirestore = async (history) => {
    try {
      const userDocRef = doc(db, 'users', activeUid);
      await setDoc(userDocRef, {
        questionnaireAnswers: history,
        lastUpdated: serverTimestamp()
      }, { merge: true });
      console.log("Questionnaire answers saved to Firestore successfully.");
    } catch (error) {
      console.error("Error saving questionnaire to Firestore:", error);
    }
  };

  const handleOptionSelect = (option) => {
    // Save answer and suggestions silently
    const currentQ = dataset.questionnaire[currentQIndex];
    const updatedHistory = [
      ...contextHistory,
      {
        questionId: currentQ.id,
        question: currentQ.question,
        optionId: option.id,
        answer: option.text,
        score: option.score,
        suggestions: option.suggestions
      }
    ];

    setContextHistory(updatedHistory);

    // Go to next instantly
    if (currentQIndex < dataset.questionnaire.length - 1) {
      setCurrentQIndex(currentQIndex + 1);
    } else {
      // Save full questionnaire to Firestore
      saveQuestionnaireToFirestore(updatedHistory);
      setPhase('chat');
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const typedText = inputText;
    setInputText("");

    try {
      // 1. Add user's message to Firestore
      await addDoc(messagesCollectionRef, {
        text: typedText,
        sender: 'user',
        timestamp: serverTimestamp()
      });

      // 2. Build history snapshot for backend including the new user message
      const historySnapshot = [...messages, { text: typedText, sender: 'user' }].map(m => ({
        sender: m.sender,
        text: m.text
      }));

      // 3. Request bot reply
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: typedText,
          contextHistory: contextHistory,
          chatHistory: historySnapshot,
          use_local_model: useLocalModel
        })
      });
      
      const data = await response.json();

      // 4. Save bot's reply to Firestore
      await addDoc(messagesCollectionRef, {
        text: data.reply || "දෝෂයකි.",
        sender: 'bot',
        timestamp: serverTimestamp()
      });
    } catch (error) {
      console.error("Chat Error:", error);
      await addDoc(messagesCollectionRef, {
        text: "ජාල දෝෂයකි. කරුණාකර නැවත උත්සාහ කරන්න.",
        sender: 'bot',
        timestamp: serverTimestamp()
      });
    }
  };

  const finishChatAndGetRecommendations = async () => {
    setPhase('summary');
    setIsLoadingSummary(true);

    try {
      const response = await fetch('http://localhost:5000/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          answers: contextHistory,
          chatHistory: messages
        })
      });
      
      const data = await response.json();
      
      let filtered = data.suggestions || [];
      
      // Add urgent hotline if score was 5 anywhere (placed at the end)
      const maxScore = Math.max(...contextHistory.map(c => c.score), 0);
      if (maxScore === 5 && !filtered.some(s => s.includes("1926"))) {
        filtered.push("🚨 කරුණාකර හැකි ඉක්මනින් මනෝවිද්‍යා උපදේශකයෙකුගේ සහාය ලබා ගන්න හෝ 1926 (ජාතික මානසික සෞඛ්‍ය උපකාරක සේවය) වෙත අමතන්න.");
      }
      
      setFinalSuggestions(filtered);
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      // Fallback
      setFinalSuggestions(["තාක්ෂණික දෝෂයක්. කරුණාකර මඳ වේලාවකින් නැවත උත්සාහ කරන්න."]);
    } finally {
      setIsLoadingSummary(false);
    }
  };

  const handleDownloadPDF = () => {
    const element = document.getElementById('pdf-content');
    const opt = {
      margin:       0.5,
      filename:     'Mental_Health_Report.pdf',
      image:        { type: 'jpeg', quality: 0.98 },
      html2canvas:  { scale: 2 },
      jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
    };
    
    // Create PDF
    html2pdf().set(opt).from(element).save();
  };

  // ---------------- Render Functions ----------------

  const renderWelcome = () => (
    <div className="hero-section">
      <h1>ඔබේ මානසික සෞඛ්‍යය මිතුරා</h1>
      <p>අපි ප්‍රශ්න කිහිපයකින් ආරම්භ කරමු. මෙහි ලබාදෙන පිළිතුරු ඉතා රහසිගත වන අතර, ඔබගේ පිළිතුරු සියල්ල සලකා බලා අවසානයේදී ඔබට ගැලපෙනම යෝජනා ලබා දෙනු ඇත.</p>
      <button className="btn-primary" onClick={handleStart}>ආරම්භ කරන්න</button>
    </div>
  );

  const renderQuestionnaire = () => {
    const questionObj = dataset.questionnaire[currentQIndex];

    return (
      <div className="question-card">
        <div className="progress-text">ප්‍රශ්නය {currentQIndex + 1} / {dataset.questionnaire.length}</div>
        <h2 className="question-text">{questionObj.question}</h2>
        
        <div className="options-container">
          {questionObj.options.map((opt, i) => (
            <button 
              key={i} 
              className="option-btn"
              onClick={() => handleOptionSelect(opt)}
            >
              {opt.text}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const renderChat = () => (
    <div className="chat-container">
      <div className="progress-text" style={{textAlign: 'center', marginBottom: '10px'}}>
        විවෘත සාකච්ඡාව (Free Chat)
        <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px', marginTop: '10px', fontSize: '14px'}}>
          <span>Gemini AI</span>
          <label className="switch">
            <input type="checkbox" checked={useLocalModel} onChange={(e) => setUseLocalModel(e.target.checked)} />
            <span className="slider round"></span>
          </label>
          <span>Local Model (30M)</span>
        </div>
      </div>
      <div className="chat-messages">
        {messages.map(msg => (
          <div key={msg.id} className={`message ${msg.sender}`}>
            {msg.text}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input-area">
        <input 
          type="text" 
          className="chat-input"
          placeholder="ඔබේ අදහස මෙහි ලියන්න..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
        />
        <button className="send-btn" onClick={handleSendMessage}>➤</button>
      </div>
      <button className="btn-primary finish-btn" onClick={finishChatAndGetRecommendations}>
        අවසන් කර යෝජනා ලබා ගන්න
      </button>
    </div>
  );

  const renderSummary = () => {
    return (
      <div className="hero-section">
        <div id="pdf-content" style={{ padding: '40px', backgroundColor: '#1e293b', color: '#ffffff', borderRadius: '8px' }}>
          <h1 style={{ color: '#38bdf8', marginBottom: '20px' }}>අවසාන යෝජනා (Top Suggestions)</h1>
          
          {isLoadingSummary ? (
            <div style={{marginTop: 30}}>
              <p>කෘතිම බුද්ධිය (AI) මගින් ඔබට වඩාත් සුදුසු යෝජනා තෝරමින් පවතී...</p>
              <div className="spinner" style={{marginTop: 20}}>Loading...</div>
            </div>
          ) : (
            <>
              <p>ඔබ ලබාදුන් පිළිතුරු AI මගින් විශ්ලේෂණය කර පහත යෝජනා ඉදිරිපත් කර ඇත:</p>
              <div className="suggestions-box" style={{textAlign: 'left', marginTop: 20}}>
                <ul className="suggestions-list">
                  {finalSuggestions.map((sugg, idx) => (
                    <li key={idx} style={{ marginBottom: '10px' }}>{sugg}</li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
        
        {!isLoadingSummary && (
          <div style={{ marginTop: 30, display: 'flex', gap: '15px', justifyContent: 'center' }}>
            <button className="btn-primary" onClick={handleDownloadPDF} style={{ backgroundColor: '#2ecc71' }}>
              වාර්තාව Download කරන්න (PDF)
            </button>
            <button className="btn-primary" onClick={() => window.location.reload()}>
              නැවත ආරම්භ කරන්න
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-container">
      {phase === 'welcome' && renderWelcome()}
      {phase === 'questionnaire' && renderQuestionnaire()}
      {phase === 'chat' && renderChat()}
      {phase === 'summary' && renderSummary()}
    </div>
  );
}

export default App;
