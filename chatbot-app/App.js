import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import WelcomeScreen from './src/screens/WelcomeScreen';
import QuestionnaireScreen from './src/screens/QuestionnaireScreen';
import ChatScreen from './src/screens/ChatScreen';
import SummaryScreen from './src/screens/SummaryScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator 
        initialRouteName="Welcome"
        screenOptions={{
          headerStyle: {
            backgroundColor: '#3498db',
          },
          headerTintColor: '#fff',
          headerTitleStyle: {
            fontWeight: 'bold',
          },
          headerTitleAlign: 'center',
        }}
      >
        <Stack.Screen 
          name="Welcome" 
          component={WelcomeScreen} 
          options={{ title: 'මානසික සෞඛ්‍යය මිතුරා' }} 
        />
        <Stack.Screen 
          name="Questionnaire" 
          component={QuestionnaireScreen} 
          options={{ title: 'ප්‍රශ්නාවලිය' }} 
        />
        <Stack.Screen 
          name="Chat" 
          component={ChatScreen} 
          options={{ title: 'සාකච්ඡාව' }} 
        />
        <Stack.Screen 
          name="Summary" 
          component={SummaryScreen} 
          options={{ title: 'අවසාන යෝජනා', headerBackVisible: false }} 
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
