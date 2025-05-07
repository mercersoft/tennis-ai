import React, { useState, useEffect, useRef, useCallback } from 'react';
// Restore original UI imports
import { ChatInput } from './ChatInput';
import { ChatMessage } from './ChatMessage';
import { ChatRole } from '../types/ChatMessage'; // Adjust path if needed
import { AgentMessage } from '../types/agent-types'; // Adjust path

// Import singleton instances and initializer

const MAX_MESSAGES = 50; // Limit message history

// Use original Message interface if different from InternalMessage used before
interface Message {
  role: ChatRole;
  content: string;
  reasoning?: string; // Keep if used by ChatMessage
  metadata?: Record<string, unknown>; // Changed any to unknown
}

interface ChatProps {
  isDarkMode: boolean; // Add back if needed by original UI
  onAgentMessage?: (message: AgentMessage) => void;
}

export const Chat: React.FC<ChatProps> = ({ isDarkMode, onAgentMessage }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! How can I help you today?',
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Dummy values for now, these will need to be handled properly or removed

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Callback ref for handling messages from agents
  // This combines the internal logic and the logic seen in the original internal handler

    // Handle setActiveItem specifically (from original code structure)

    // Add agent messages to display (optional, might duplicate info shown by parent)
    // Consider only forwarding or displaying specific types like reasoning/action

    // Forward the message if an external handler is provided

  // Effect to initialize agents when model and callbacks are ready

  // handleSendMessage using singleton instances and original Message structure
  const handleSendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: text
    };

    setMessages(prev => [...prev.slice(-MAX_MESSAGES + 1), userMessage]);
    setIsLoading(true);
    setError(null);

    if (onAgentMessage) {
      onAgentMessage({
        type: 'action',
        agentName: 'Chat',
        payload: `Starting new query: "${text}"`,
        isNewQuery: true,
        timestamp: Date.now()
      });
    }

    const chatUrl = import.meta.env.VITE_CHAT_URL;
    if (!chatUrl) {
      setError('Chat URL is not configured. Please set VITE_CHAT_URL in your environment.');
      setIsLoading(false);
      return;
    }

    try {
      const response = await fetch(chatUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: text }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ message: 'Failed to get error details' }));
        throw new Error(`Server error: ${response.status} ${response.statusText}. ${errorData.message || ''}`);
      }

      const data = await response.json();
      
      // Assuming the server responds with a JSON object that has a 'reply' field
      const assistantMessageContent = data.answer || "Sorry, I didn't get a valid response.";

      const assistantMessage: Message = {
        role: 'assistant',
        content: assistantMessageContent,
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err: unknown) {
      console.error("Error sending message:", err);
      if (err instanceof Error) {
        setError(err.message);
      } else if (typeof err === 'string') {
        setError(err);
      } else {
        setError('Failed to send message. Please try again.');
      }
    } finally {
      setIsLoading(false);
    }

  }, [isLoading, onAgentMessage]);

  // Render using original structure
  return (
    <div className="h-full w-full flex flex-col">
      <div className="flex-1 overflow-y-auto">
        <div className="p-4 space-y-4">
          {messages.map((msg, index) => (
            <ChatMessage
              key={index}
              role={msg.role}
              content={msg.content}
              reasoning={msg.reasoning}
              isDarkMode={isDarkMode} // Pass prop back in
            />
          ))}
          {isLoading && (
            <div className="flex justify-center items-center py-4">
              {/* Use a simple spinner or original loading indicator */}
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
            </div>
          )}
          {error && (
            <div className="text-red-500 text-center py-4">
              Error: {error}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      <div className="flex-shrink-0 border-t p-4">
        {/* Pass isDarkMode back to ChatInput, remove isLoading */}
        <ChatInput isDarkMode={isDarkMode} onSendMessage={handleSendMessage} />
      </div>
    </div>
  );
}; 