import { useState, useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition as useSR } from 'react-speech-recognition';

interface UseSpeechRecognitionReturn {
  transcript: string;
  listening: boolean;
  error: string | null;
  startListening: () => void;
  stopListening: () => void;
  resetTranscript: () => void;
  browserSupportsSpeechRecognition: boolean;
}

interface UseSpeechRecognitionProps {
  onSpeechEnd?: (transcript: string) => void;
}

export const useSpeechRecognition = ({ onSpeechEnd }: UseSpeechRecognitionProps = {}): UseSpeechRecognitionReturn => {
  const [error, setError] = useState<string | null>(null);
  
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSR();

  useEffect(() => {
    if (!browserSupportsSpeechRecognition) {
      setError('Your browser does not support speech recognition. Please use Chrome, Edge, or Safari.');
    }
  }, [browserSupportsSpeechRecognition]);

  // Handle speech end and submit text
  useEffect(() => {
    if (!listening && transcript && onSpeechEnd) {
      const finalTranscript = transcript;
      resetTranscript(); // Clear the transcript before calling onSpeechEnd
      onSpeechEnd(finalTranscript);
    }
  }, [listening, transcript, onSpeechEnd, resetTranscript]);

  const startListening = async () => {
    try {
      setError(null);
      resetTranscript(); // Clear any previous transcript when starting
      
      await SpeechRecognition.startListening({ 
        continuous: true,  // Keep listening until explicitly stopped
        interimResults: true  // Get results while speaking
      });
    } catch (err) {
      setError('Failed to start speech recognition. Please check your microphone permissions.');
      console.error('Speech recognition error:', err);
    }
  };

  const stopListening = async () => {
    try {
      await SpeechRecognition.stopListening();
    } catch (err) {
      setError('Failed to stop speech recognition.');
      console.error('Speech recognition error:', err);
    }
  };

  return {
    transcript,
    listening,
    error,
    startListening,
    stopListening,
    resetTranscript,
    browserSupportsSpeechRecognition
  };
}; 