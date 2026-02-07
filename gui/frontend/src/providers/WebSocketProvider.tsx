/**
 * WebSocket Provider
 * Phase 6.1 - Core Infrastructure
 *
 * Initializes and maintains the global WebSocket connection.
 * Wraps the app to ensure connection persists across route changes.
 */

import { createContext, useContext, ReactNode } from 'react';
import { useGlobalWebSocket } from '@/hooks/useWebSocket';

type MessageHandler = (data: unknown) => void;

interface WebSocketContextValue {
  isConnected: boolean;
  connectionId: string | null;
  subscribe: (channel: string, handler: MessageHandler) => () => void;
  unsubscribe: (channel: string) => void;
  send: (message: object) => void;
  connect: () => void;
  disconnect: () => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
}

export function WebSocketProvider({ children }: WebSocketProviderProps) {
  const ws = useGlobalWebSocket();

  return <WebSocketContext.Provider value={ws}>{children}</WebSocketContext.Provider>;
}

export function useWebSocketContext(): WebSocketContextValue {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocketContext must be used within a WebSocketProvider');
  }
  return context;
}
