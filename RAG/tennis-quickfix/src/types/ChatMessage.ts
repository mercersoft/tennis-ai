export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  text: string;
  timestamp?: Date; // Optional timestamp for when the message was sent
}

// Helper function to create a new chat message
export const createChatMessage = (role: ChatRole, text: string): ChatMessage => ({
  role,
  text,
  timestamp: new Date()
}); 