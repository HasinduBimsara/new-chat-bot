import { initializeApp, getApps, getApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

// Firebase configuration placeholder. 
// Replace these with your actual Firebase project configuration from Firebase Console.
const firebaseConfig = {
  apiKey: "AIzaSyD4sM4TspMdhDq38XF0rVkgkpm7q_HpfO8",
  authDomain: "chatbot-testing-4a7f9.firebaseapp.com",
  projectId: "chatbot-testing-4a7f9",
  storageBucket: "chatbot-testing-4a7f9.firebasestorage.app",
  messagingSenderId: "182721116114",
  appId: "1:182721116114:web:68ac207b2789d63a817a0b"
};

// Initialize Firebase App
const app = getApps().length === 0 ? initializeApp(firebaseConfig) : getApp();

// Initialize Auth
const auth = getAuth(app);

// Initialize Firestore
const db = getFirestore(app);

export { app, auth, db };
