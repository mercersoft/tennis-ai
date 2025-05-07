import React, { useState } from 'react';
import { MicrophoneButton } from './MicrophoneButton';

interface ChatInputProps {
  isDarkMode: boolean;
  onSendMessage?: (message: string) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({ isDarkMode, onSendMessage }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSend = () => {
    if (inputValue.trim() && onSendMessage) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTranscriptChange = (transcript: string) => {
    setInputValue(transcript);
  };

  const handleSpeechEnd = (transcript: string) => {
    if (transcript.trim() && onSendMessage) {
      onSendMessage(transcript.trim());
      setInputValue('');
    }
  };

  return (
    <div className="flex items-center gap-2 p-4">
      <MicrophoneButton 
        isDarkMode={isDarkMode}
        onTranscriptChange={handleTranscriptChange}
        onSpeechEnd={handleSpeechEnd}
      />

      {/* Input Field */}
      <div className="flex-1 relative">
        <input
          type="text"
          placeholder="Ask a question..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          className={`
            w-full
            py-2 px-4
            rounded-full
            outline-none
            transition-colors
            ${isDarkMode 
              ? 'bg-aws-bg-dark text-aws-text-primary placeholder-aws-text-secondary border-aws-border-dark' 
              : 'bg-gray-100 text-gray-900 placeholder-gray-500 border-gray-200'}
            border
            focus:ring-2
            focus:ring-blue-500
            focus:border-transparent
          `}
        />
        {/* Send Button */}
        <button
          onClick={handleSend}
          className={`
            absolute right-2 top-1/2 -translate-y-1/2
            p-1.5
            rounded-full
            transition-colors
            ${isDarkMode 
              ? 'text-aws-text-primary hover:bg-aws-bg-light' 
              : 'text-blue-600 hover:bg-gray-200'}
          `}
          aria-label="Send message"
        >
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            viewBox="0 0 24 24" 
            fill="currentColor" 
            className="w-5 h-5"
          >
            <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
          </svg>
        </button>
      </div>
    </div>
  );
}; 