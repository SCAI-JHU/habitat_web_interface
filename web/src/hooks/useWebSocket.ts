import { useEffect, useRef, useCallback } from 'react';

interface UseWebSocketProps {
  onMessage: (message: string) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: () => void;
}

export const useWebSocket = ({ onMessage, onOpen, onClose, onError }: UseWebSocketProps) => {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>();
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    // Stop reconnecting after max attempts
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.log('Max WebSocket reconnection attempts reached. Giving up.');
      return;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${window.location.host}/ws/live_feed`;

    console.log(`Connecting to WebSocket: ${wsUrl} (attempt ${reconnectAttemptsRef.current + 1})`);
    
    try {
      const socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        console.log('WebSocket connection established.');
        reconnectAttemptsRef.current = 0; // Reset on successful connection
        onOpen?.();
      };

      socket.onmessage = (event) => {
        onMessage(event.data);
      };

      socket.onclose = (event) => {
        console.log('WebSocket connection closed.', event);
        onClose?.();
        socketRef.current = null;

        // Exponential backoff: 5s, 10s, 20s, 40s, 80s
        const delay = Math.min(5000 * Math.pow(2, reconnectAttemptsRef.current), 80000);
        reconnectAttemptsRef.current++;

        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(`Attempting to reconnect in ${delay}ms...`);
            connect();
          }, delay);
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.();
      };

      socketRef.current = socket;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      reconnectAttemptsRef.current++;
    }
  }, [onMessage, onOpen, onClose, onError]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [connect]);

  return {
    socket: socketRef.current,
    isConnected: socketRef.current?.readyState === WebSocket.OPEN
  };
};

