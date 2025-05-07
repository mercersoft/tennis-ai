import React from 'react';
import { ChatRole } from '../types/ChatMessage';
import ReactMarkdown from 'react-markdown';
import { User } from 'lucide-react';

interface ChatMessageProps {
  role: ChatRole;
  content: string;
  reasoning?: string;
  isDarkMode: boolean;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ role, content, reasoning, isDarkMode }) => {
  const isUser = role === 'user';
  
  return (
    <div className="flex items-start gap-3">
      <div className={`
        flex-shrink-0 w-8 h-8 rounded-full
        flex items-center justify-center
        self-center
        ${isUser 
          ? (isDarkMode ? 'bg-blue-600' : 'bg-blue-500')
          : 'bg-aws-header'
        }
      `}>
        {isUser ? (
          <User className="w-5 h-5 text-white" />
        ) : (
          <img 
            src="/tennis-ai.svg" 
            alt="Tennis AI Coach Logo" 
            className="w-7 h-7"
          />
        )}
      </div>
      <div className={`
        flex-1 px-4 py-2 rounded-lg
        ${isUser ? 'rounded-tr-none' : 'rounded-tl-none'}
        ${isDarkMode 
          ? (isUser ? 'bg-blue-600' : 'bg-blue-500/30') 
          : (isUser ? 'bg-blue-500' : 'bg-blue-100')}
        ${isDarkMode 
          ? 'text-white' 
          : (isUser ? 'text-white' : 'text-blue-900')}
      `}>
        <ReactMarkdown 
          components={{
            ol: ({ children }) => (
              <ol className="list-decimal list-inside space-y-1">
                {children}
              </ol>
            ),
            li: ({ children }) => (
              <li className="pl-2">{children}</li>
            )
          }}
        >
          {content}
        </ReactMarkdown>
        {reasoning && !isUser && (
          <div className="mt-2 pt-2 border-t border-blue-200 text-sm opacity-75">
            <ReactMarkdown 
              components={{
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside space-y-1">
                    {children}
                  </ol>
                ),
                li: ({ children }) => (
                  <li className="pl-2">{children}</li>
                )
              }}
            >
              {reasoning}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}; 