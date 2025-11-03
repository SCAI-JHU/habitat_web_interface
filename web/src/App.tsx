import { useState, useCallback, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { LiveFeed } from './components/LiveFeed';
import { Metrics } from './components/Metrics';
import { Terminal } from './components/Terminal';
import { SystemLogs } from './components/SystemLogs';
import { useWebSocket } from './hooks/useWebSocket';
import { getCurrentTime, generateId } from './utils/helpers';
import { AppState, StatusType, MessageType, LogLevel, TerminalLine, SystemLogLine } from './types';

const INITIAL_METRICS = Array(20).fill(0);

function App() {
  const [state, setState] = useState<AppState>({
    status: 'Idle',
    statusType: 'idle',
    imageSrc: null,
    isRunning: false,
    terminalLines: [],
    systemLogs: [],
    metrics: {
      cpu: INITIAL_METRICS,
      memory: INITIAL_METRICS
    }
  });

  const imagePollingIntervalRef = useRef<ReturnType<typeof setInterval>>();
  const mockMetricsIntervalRef = useRef<ReturnType<typeof setInterval>>();

  // Helper functions
  const addTerminalLine = useCallback((message: string, type: MessageType = 'info') => {
    const newLine: TerminalLine = {
      id: generateId(),
      timestamp: getCurrentTime(),
      message,
      type
    };
    setState(prev => ({
      ...prev,
      terminalLines: [...prev.terminalLines, newLine]
    }));
  }, []);

  const addSystemLog = useCallback((message: string, level: LogLevel = 'INFO') => {
    const newLog: SystemLogLine = {
      id: generateId(),
      timestamp: getCurrentTime(),
      message,
      level
    };
    setState(prev => ({
      ...prev,
      systemLogs: [...prev.systemLogs, newLog]
    }));
  }, []);

  const setStatus = useCallback((message: string, type: StatusType = 'info') => {
    setState(prev => ({
      ...prev,
      status: message,
      statusType: type
    }));
  }, []);

  const updateImageDisplay = useCallback((imageSrc: string | null) => {
    setState(prev => ({
      ...prev,
      imageSrc
    }));
  }, []);

  const updateMetric = useCallback((type: 'cpu' | 'memory', value: number) => {
    setState(prev => ({
      ...prev,
      metrics: {
        ...prev.metrics,
        [type]: [...prev.metrics[type].slice(-19), value]
      }
    }));
  }, []);

  // WebSocket message handler
  const handleSocketMessage = useCallback((message: string) => {
    try {
      if (message.startsWith('data:image/')) {
        updateImageDisplay(message);
      } else if (message.startsWith('status:')) {
        const [, statusType, statusMessage] = message.split(':', 3);
        console.log(`Status Update: ${statusType} - ${statusMessage}`);
        setStatus(`${statusType} - ${statusMessage}`, statusType as StatusType);
        addTerminalLine(`[${statusType.toUpperCase()}] ${statusMessage}`, statusType as MessageType);

        if (statusType === 'complete' || statusType === 'error') {
          setState(prev => ({ ...prev, isRunning: false }));
          if (imagePollingIntervalRef.current) {
            clearInterval(imagePollingIntervalRef.current);
          }
        }
      } else if (message.startsWith('log:')) {
        const [, level, logMessage] = message.split(':', 3);
        addSystemLog(logMessage, level.toUpperCase() as LogLevel);
      } else if (message.startsWith('metric:')) {
        const [, metricType, metricValue] = message.split(':', 3);
        const value = parseFloat(metricValue);
        if (metricType === 'cpu') {
          updateMetric('cpu', value);
        } else if (metricType === 'mem') {
          updateMetric('memory', value);
        }
      } else {
        console.log('Received unhandled message:', message);
        addSystemLog(`Unhandled message: ${message}`, 'DEBUG');
      }
    } catch (e) {
      console.error('Error handling message:', e);
      addSystemLog(`Failed to parse message: ${message}`, 'ERROR');
    }
  }, [addTerminalLine, addSystemLog, setStatus, updateImageDisplay, updateMetric]);

  // WebSocket setup (optional - for real-time updates)
  useWebSocket({
    onMessage: handleSocketMessage,
    onOpen: () => {
      addTerminalLine('WebSocket connected - Real-time updates enabled.', 'success');
      addSystemLog('WebSocket connected successfully.', 'INFO');
      setStatus('Connected. Ready to run.', 'info');
    },
    onClose: () => {
      // Silent on close - WebSocket through SSH tunnels is unreliable
      // The app will work fine with polling instead
      console.log('WebSocket closed - using polling mode instead');
      setState(prev => ({ ...prev, isRunning: false }));
      if (imagePollingIntervalRef.current) {
        clearInterval(imagePollingIntervalRef.current);
      }
    },
    onError: () => {
      // Silent on error - not critical for functionality
      console.log('WebSocket error - continuing without real-time updates');
    }
  });

  // Fetch latest image
  const fetchLatestImage = useCallback(async () => {
    try {
      const response = await fetch('/latest-image');
      const data = await response.json();

      if (data.image) {
        updateImageDisplay(data.image);
        addTerminalLine(`Displaying ${data.filename} (${data.total_images} total images)`, 'info');
      } else {
        updateImageDisplay(null);
      }
    } catch (error) {
      console.error('Failed to fetch latest image:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addTerminalLine(`Failed to fetch latest image: ${errorMessage}`, 'error');
      addSystemLog(`Failed to fetch latest image: ${errorMessage}`, 'ERROR');
    }
  }, [addTerminalLine, addSystemLog, updateImageDisplay]);

  // Handle run simulation
  const handleRunSimulation = useCallback(async () => {
    console.log('Requesting simulation start...');
    addTerminalLine('Starting simulation...', 'info');
    addSystemLog('Simulation run requested by user.', 'INFO');
    setState(prev => ({ ...prev, isRunning: true }));
    setStatus('Requesting simulation start...', 'running');
    updateImageDisplay(null);

    try {
      const response = await fetch('/run-simulation', { method: 'POST' });
      const data = await response.json();
      console.log('Simulation start request sent:', data.message);
      addTerminalLine(`Simulation start request sent: ${data.message}`, 'success');
      addSystemLog(`Server responded: ${data.message}`, 'INFO');

      if (imagePollingIntervalRef.current) {
        clearInterval(imagePollingIntervalRef.current);
      }
      imagePollingIntervalRef.current = setInterval(fetchLatestImage, 500);
    } catch (error) {
      console.error('Failed to send start simulation request:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addTerminalLine(`Failed to send start simulation request: ${errorMessage}`, 'error');
      addSystemLog(`Failed to start simulation: ${errorMessage}`, 'ERROR');
      setStatus('Error starting simulation.', 'error');
      setState(prev => ({ ...prev, isRunning: false }));
    }
  }, [addTerminalLine, addSystemLog, setStatus, updateImageDisplay, fetchLatestImage]);

  // Handle clear terminal
  const handleClearTerminal = useCallback(() => {
    setState(prev => ({
      ...prev,
      terminalLines: []
    }));
    addTerminalLine('Terminal cleared.', 'info');
  }, [addTerminalLine]);

  // Handle clear logs
  const handleClearLogs = useCallback(() => {
    setState(prev => ({
      ...prev,
      systemLogs: []
    }));
    addSystemLog('System logs cleared.', 'INFO');
  }, [addSystemLog]);

  // Initialize app
  useEffect(() => {
    addTerminalLine('Initializing system...', 'info');
    addSystemLog('Control panel UI initialized.', 'INFO');
    addSystemLog('Metrics charts initialized.', 'DEBUG');
    addTerminalLine('Loading latest image...', 'info');
    fetchLatestImage();

    // Mock metrics for demo
    addSystemLog('Starting mock metrics for demo. Remove in production.', 'WARN');
    mockMetricsIntervalRef.current = setInterval(() => {
      if (!state.isRunning) {
        updateMetric('cpu', Math.random() * 5);
        updateMetric('memory', Math.random() * 3 + 10);
      }
    }, 1000);

    return () => {
      if (imagePollingIntervalRef.current) {
        clearInterval(imagePollingIntervalRef.current);
      }
      if (mockMetricsIntervalRef.current) {
        clearInterval(mockMetricsIntervalRef.current);
      }
    };
  }, []);

  return (
    <div className="font-sans m-0 bg-gray-900 text-gray-200 flex h-screen overflow-hidden">
      <Sidebar
        onRunSimulation={handleRunSimulation}
        onRefreshImage={fetchLatestImage}
        onClearTerminal={handleClearTerminal}
        onClearLogs={handleClearLogs}
        isRunning={state.isRunning}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        <Header status={state.status} statusType={state.statusType} />

        <div className="flex-1 p-4 grid grid-cols-1 lg:grid-cols-5 gap-4 overflow-y-auto custom-scrollbar">
          <div className="lg:col-span-3 flex flex-col space-y-4">
            <LiveFeed imageSrc={state.imageSrc} />
            <Metrics metrics={state.metrics} />
          </div>

          <div className="lg:col-span-2 flex flex-col space-y-4 h-full">
            <Terminal lines={state.terminalLines} />
            <SystemLogs logs={state.systemLogs} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;

