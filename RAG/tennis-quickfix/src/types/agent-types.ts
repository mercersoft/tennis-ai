/**
 * Represents a message emitted by an agent, which can be either reasoning or an action.
 */
export interface AgentMessage {
  type: 'reasoning' | 'action' | 'setActiveItem';
  agentName: string; // Name of the agent emitting the message
  payload: string; // The core message content (reasoning text or action details)
  isNewQuery?: boolean; // Optional flag for actions indicating a new query start
  timestamp: number;
}
