import React from 'react';
import { useSpeechRecognition } from '../hooks/useSpeechRecognition';

interface MicrophoneButtonProps {
  isDarkMode: boolean;
  onTranscriptChange: (transcript: string) => void;
  onSpeechEnd?: (transcript: string) => void;
}

export const MicrophoneButton: React.FC<MicrophoneButtonProps> = ({ 
  isDarkMode,
  onTranscriptChange,
  onSpeechEnd
}) => {
  const {
    listening,
    error,
    startListening,
    stopListening,
    transcript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition({ onSpeechEnd });

  React.useEffect(() => {
    if (transcript) {
      onTranscriptChange(transcript);
    }
  }, [transcript, onTranscriptChange]);

  // Log final transcript when speech recognition stops
  React.useEffect(() => {
    if (!listening && transcript) {
      console.log('Final transcript:', transcript);
    }
  }, [listening, transcript]);

  if (!browserSupportsSpeechRecognition) {
    return null;
  }

  const handleClick = () => {
    if (listening) {
      stopListening();
    } else {
      startListening();
    }
  };

  return (
    <button
      onClick={handleClick}
      className={`
        flex items-center justify-center
        w-10 h-10
        rounded-full
        transition-colors
        ${isDarkMode 
          ? 'bg-aws-bg-dark hover:bg-aws-bg-light text-aws-text-primary' 
          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'}
        ${listening ? 'animate-pulse' : ''}
      `}
      aria-label={listening ? "Stop listening" : "Start listening"}
      disabled={!!error}
    >
      <svg 
        xmlns="http://www.w3.org/2000/svg" 
        viewBox="0 0 24 24" 
        fill="currentColor" 
        className="w-5 h-5"
      >
        <path d="M8.25 4.5a3.75 3.75 0 117.5 0v8.25a3.75 3.75 0 11-7.5 0V4.5z" />
        <path d="M6 10.5a.75.75 0 01.75.75v1.5a5.25 5.25 0 1010.5 0v-1.5a.75.75 0 011.5 0v1.5a6.751 6.751 0 01-6 6.709v2.291h3a.75.75 0 010 1.5h-7.5a.75.75 0 010-1.5h3v-2.291a6.751 6.751 0 01-6-6.709v-1.5A.75.75 0 016 10.5z" />
      </svg>
    </button>
  );
}; 