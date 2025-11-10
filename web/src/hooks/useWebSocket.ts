import { useEffect, useRef, useCallback } from 'react';
import { WS_HOST } from '../config';

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

  // Store callbacks in refs to prevent them from being dependencies
  // This ensures `connect` function is stable and doesn't change on re-renders
  const onMessageRef = useRef(onMessage);
  const onOpenRef = useRef(onOpen);
  const onCloseRef = useRef(onClose);
  const onErrorRef = useRef(onError);

  // Update refs when props change
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    onOpenRef.current = onOpen;
  }, [onOpen]);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);


  const connect = useCallback(() => {
    // --- FIX: Add a guard to prevent multiple connections ---
    if (socketRef.current && socketRef.current.readyState < 2) {
      // 0 = CONNECTING, 1 = OPEN
      console.log("WebSocket already connecting or open. Skipping new connection.");
      return;
    }
    // ---------------------------------------------------------

    // Stop reconnecting after max attempts
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.log('Max WebSocket reconnection attempts reached. Giving up.');
      onCloseRef.current?.(); // Call onClose when giving up
      return;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${WS_HOST}/ws/live_feed`;

    console.log(`Connecting to WebSocket: ${wsUrl} (attempt ${reconnectAttemptsRef.current + 1})`);
    
    try {
      const socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        console.log('WebSocket connection established.');
        reconnectAttemptsRef.current = 0; // Reset on successful connection
        onOpenRef.current?.();
      };

      socket.onmessage = (event) => {
        onMessageRef.current(event.data);
      };

      socket.onclose = (event) => {
        console.log('WebSocket connection closed.', event);
        socketRef.current = null;

        // Don't call onCloseRef.current here, only when giving up
        // or it will trigger polling mode indefinitely

        // Exponential backoff: 2s, 4s, 8s, 16s, 32s
        const delay = Math.min(2000 * Math.pow(2, reconnectAttemptsRef.current), 32000);
        reconnectAttemptsRef.current++;

        if (reconnectAttemptsRef.current <= maxReconnectAttempts) {
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(`Attempting to reconnect in ${delay}ms...`);
            connect();
          }, delay);
        } else {
          console.log('Max WebSocket reconnection attempts reached. Giving up.');
          onCloseRef.current?.(); // Call onClose now that we've given up
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        onErrorRef.current?.();
        // Don't auto-reconnect on error, onclose will handle it
      };

      socketRef.current = socket;
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      reconnectAttemptsRef.current++;
      // Retry after a delay if creation itself failed
      setTimeout(connect, 2000);
    }
  }, []); // --- FIX: Removed all dependencies ---

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socketRef.current) {
        console.log("Cleaning up WebSocket on component unmount.");
        socketRef.current.onclose = null; // Prevent reconnect logic from firing on unmount
        socketRef.current.close();
      }
    };
  }, [connect]); // --- FIX: This should now depend on the stable `connect` function ---
                 // (which itself has no dependencies)

  return {
    socket: socketRef.current,
    isConnected: socketRef.current?.readyState === WebSocket.OPEN
  };
};