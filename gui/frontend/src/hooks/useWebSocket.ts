/**
 * WebSocket Hook
 * Phase 6.1 - Core Infrastructure
 *
 * Manages WebSocket connection with auto-reconnect and channel subscriptions.
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { useUIStore } from '@/stores/uiStore';

type MessageHandler = (data: unknown) => void;

interface WebSocketMessage {
  type: string;
  channel?: string;
  data?: unknown;
  message?: string;
  connection_id?: string;
  timestamp?: string;
}

interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  connectionId: string | null;
  subscribe: (channel: string, handler: MessageHandler) => () => void;
  unsubscribe: (channel: string) => void;
  send: (message: object) => void;
  connect: () => void;
  disconnect: () => void;
}

// Compute the WebSocket URL based on environment
function getDefaultWsUrl(): string {
  const hostname = window.location.hostname;

  // Lightning AI cloudspaces: frontend is 5173-xxx, backend is 8000-xxx
  if (hostname.includes('.cloudspaces.litng.ai')) {
    // Replace 5173- prefix with 8000-
    const backendHost = hostname.replace(/^5173-/, '8000-');
    // Use wss:// for secure connection on Lightning AI
    return `wss://${backendHost}/ws`;
  }

  // Local development: same hostname, different port
  return `ws://${hostname}:8000/ws`;
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    url = getDefaultWsUrl(),
    autoConnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef<Map<string, Set<MessageHandler>>>(new Map());
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const [isConnected, setIsConnected] = useState(false);
  const [connectionId, setConnectionId] = useState<string | null>(null);

  const { setWsConnected } = useUIStore();

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('[WebSocket] Connected');
        setIsConnected(true);
        setWsConnected(true);
        reconnectAttemptsRef.current = 0;
      };

      ws.onclose = (event) => {
        console.log('[WebSocket] Disconnected', event.code, event.reason);
        setIsConnected(false);
        setWsConnected(false);
        setConnectionId(null);

        // Auto-reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++;
          console.log(
            `[WebSocket] Reconnecting in ${reconnectInterval}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`
          );
          reconnectTimeoutRef.current = setTimeout(connect, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleMessage(message);
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('[WebSocket] Connection error:', error);
    }
  }, [url, reconnectInterval, maxReconnectAttempts, setWsConnected]);

  // Handle incoming messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    // Handle connection confirmation
    if (message.type === 'connected') {
      setConnectionId(message.connection_id ?? null);
      console.log('[WebSocket] Connection ID:', message.connection_id);
      return;
    }

    // Handle subscription confirmations
    if (message.type === 'subscribed' || message.type === 'unsubscribed') {
      console.log(`[WebSocket] ${message.type} to channel:`, message.channel);
      return;
    }

    // Handle pong
    if (message.type === 'pong') {
      return;
    }

    // Route to channel handlers
    if (message.channel) {
      const handlers = handlersRef.current.get(message.channel);
      if (handlers) {
        handlers.forEach((handler) => {
          try {
            handler(message.data);
          } catch (e) {
            console.error(`[WebSocket] Handler error for ${message.channel}:`, e);
          }
        });
      }
    }

    // Also check for wildcard handlers
    const wildcardHandlers = handlersRef.current.get('*');
    if (wildcardHandlers) {
      wildcardHandlers.forEach((handler) => {
        try {
          handler(message);
        } catch (e) {
          console.error('[WebSocket] Wildcard handler error:', e);
        }
      });
    }
  }, []);

  // Disconnect
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setWsConnected(false);
  }, [setWsConnected]);

  // Subscribe to a channel
  const subscribe = useCallback(
    (channel: string, handler: MessageHandler): (() => void) => {
      // Add to local handlers
      if (!handlersRef.current.has(channel)) {
        handlersRef.current.set(channel, new Set());
      }
      handlersRef.current.get(channel)!.add(handler);

      // Send subscribe message to server
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'subscribe', channel }));
      }

      // Return unsubscribe function
      return () => {
        const handlers = handlersRef.current.get(channel);
        if (handlers) {
          handlers.delete(handler);
          if (handlers.size === 0) {
            handlersRef.current.delete(channel);
            // Send unsubscribe message
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({ type: 'unsubscribe', channel }));
            }
          }
        }
      };
    },
    []
  );

  // Unsubscribe from a channel
  const unsubscribe = useCallback((channel: string) => {
    handlersRef.current.delete(channel);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'unsubscribe', channel }));
    }
  }, []);

  // Send a message
  const send = useCallback((message: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      console.warn('[WebSocket] Cannot send - not connected');
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  // Heartbeat / ping
  useEffect(() => {
    if (!isConnected) return;

    const pingInterval = setInterval(() => {
      send({ type: 'ping', timestamp: Date.now() });
    }, 30000);

    return () => clearInterval(pingInterval);
  }, [isConnected, send]);

  return {
    isConnected,
    connectionId,
    subscribe,
    unsubscribe,
    send,
    connect,
    disconnect,
  };
}

// Singleton hook for global WebSocket instance
let globalWsInstance: UseWebSocketReturn | null = null;

export function useGlobalWebSocket(): UseWebSocketReturn {
  const ws = useWebSocket({ autoConnect: true });

  useEffect(() => {
    globalWsInstance = ws;
    return () => {
      globalWsInstance = null;
    };
  }, [ws]);

  return ws;
}

export function getGlobalWebSocket(): UseWebSocketReturn | null {
  return globalWsInstance;
}
