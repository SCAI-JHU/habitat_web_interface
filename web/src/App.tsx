import { useState, useCallback, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { LiveFeed } from './components/LiveFeed';
import { Metrics } from './components/Metrics';
import { Terminal } from './components/Terminal';
import { SystemLogs } from './components/SystemLogs';
import { RobotControl } from './components/RobotControl';
import { useWebSocket } from './hooks/useWebSocket';
import { getCurrentTime, generateId } from './utils/helpers';
import { AppState, StatusType, MessageType, LogLevel, TerminalLine, SystemLogLine } from './types';
import { API_BASE } from './config';

const INITIAL_METRICS = Array(20).fill(0);
const MOVEMENT_PATH_MAP: Record<string, string> = {
  forward: 'forward',
  backward: 'back',
  left: 'left',
  right: 'right',
  stop: 'stop',
};

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

  // --- FIX: Wrap all state setters in useCallback ---

  const addTerminalLine = useCallback((message: string, type: MessageType = 'info') => {
    const newLine: TerminalLine = {
      id: generateId(),
      timestamp: getCurrentTime(),
      message,
      type
    };
    setState(prev => ({
      ...prev,
      terminalLines: [...prev.terminalLines.slice(-100), newLine] // Prune old lines
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
      systemLogs: [...prev.systemLogs.slice(-100), newLog] // Prune old lines
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

  // --- FIX: All dependencies of handleSocketMessage are now stable ---
  const handleSocketMessage = useCallback((message: string) => {
    try {
      if (message.startsWith('data:image/')) {
        updateImageDisplay(message);
      } else if (message.startsWith('status:')) {
        const [, statusType, statusMessage] = message.split(':', 3);
        console.log(`Status Update: ${statusType} - ${statusMessage}`);
        setStatus(`${statusType} - ${statusMessage}`, statusType as StatusType);
        addTerminalLine(`[${statusType.toUpperCase()}] ${statusMessage}`, statusType as MessageType);

        // Update isRunning based on status
        if (statusType === 'running') {
          setState(prev => ({ ...prev, isRunning: true }));
        } else if (statusType === 'complete' || statusType === 'error') {
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
        // Don't log *every* unhandled message, it's spammy
        // console.log('Received unhandled message:', message);
      }
    } catch (e) {
      console.error('Error handling message:', e);
      addSystemLog(`Failed to parse message: ${message}`, 'ERROR');
    }
  }, [addTerminalLine, addSystemLog, setStatus, updateImageDisplay, updateMetric]);
  
  const buildUrl = useCallback(
    (path: string) => (API_BASE ? `${API_BASE}${path}` : path),
    []
  );

  // Fetch latest image
  const fetchLatestImage = useCallback(async () => {
    try {
      const response = await fetch(buildUrl('/latest-image'));
      if (!response.ok) {
        throw new Error(`Server responded ${response.status}`);
      }
      const data = await response.json();

      if (data.image) {
        updateImageDisplay(data.image);
      } else {
        updateImageDisplay(null);
      }
    } catch (error) {
      console.error('Failed to fetch latest image:', error);
    }
  }, [updateImageDisplay, buildUrl]);


  // WebSocket setup
  useWebSocket({
    onMessage: handleSocketMessage,
    onOpen: () => {
      addTerminalLine('WebSocket connected - Real-time updates enabled.', 'success');
      addSystemLog('WebSocket connected successfully.', 'INFO');
      setStatus('Connected. Ready to run.', 'info');
      // Stop polling once WebSocket is connected
      if (imagePollingIntervalRef.current) {
        clearInterval(imagePollingIntervalRef.current);
      }
    },
    onClose: () => {
      addSystemLog('WebSocket closed. Falling back to polling.', 'WARN');
      // Fallback to polling if WebSocket closes
      if (imagePollingIntervalRef.current) {
        clearInterval(imagePollingIntervalRef.current);
      }
      imagePollingIntervalRef.current = setInterval(fetchLatestImage, 1000); // Poll every 1s
    },
    onError: () => {
      addSystemLog('WebSocket error. Falling back to polling.', 'ERROR');
    }
  });

  // Handle run simulation
  const handleRunSimulation = useCallback(async () => {
    console.log('Requesting simulation start...');
    addTerminalLine('Starting simulation...', 'info');
    addSystemLog('Simulation run requested by user.', 'INFO');
    setState(prev => ({ ...prev, isRunning: true }));
    setStatus('Requesting simulation start...', 'running');
    updateImageDisplay(null);

    // Stop polling (if any) when we try to run
    if (imagePollingIntervalRef.current) {
      clearInterval(imagePollingIntervalRef.current);
    }

    try {
      const response = await fetch(buildUrl('/run-simulation'), { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to start");

      console.log('Simulation start request sent:', data.message);
      addTerminalLine(`Simulation start request sent: ${data.message}`, 'success');
      addSystemLog(`Server responded: ${data.message}`, 'INFO');

      // Start polling *after* simulation is confirmed running
      // WebSocket 'onOpen' will clear this if it connects
      imagePollingIntervalRef.current = setInterval(fetchLatestImage, 500);
    } catch (error) {
      console.error('Failed to send start simulation request:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addTerminalLine(`Failed to send start simulation request: ${errorMessage}`, 'error');
      addSystemLog(`Failed to start simulation: ${errorMessage}`, 'ERROR');
      setStatus('Error starting simulation.', 'error');
      setState(prev => ({ ...prev, isRunning: false }));
    }
  }, [addTerminalLine, addSystemLog, setStatus, updateImageDisplay, fetchLatestImage, buildUrl]);

  const handleStopSimulation = useCallback(async () => {
    console.log('Requesting simulation stop...');
    addTerminalLine('Stopping simulation...', 'info');
    addSystemLog('Simulation stop requested by user.', 'INFO');

    // Stop polling
    if (imagePollingIntervalRef.current) {
      clearInterval(imagePollingIntervalRef.current);
    }

    try {
      const response = await fetch(buildUrl('/stop-simulation'), { method: 'POST' });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || response.statusText);
      }
      addTerminalLine(`Simulation stop request: ${data.message}`, 'info');
      addSystemLog(`Server responded: ${data.message}`, 'INFO');
      setStatus('Simulation stopped.', 'info');
      setState(prev => ({ ...prev, isRunning: false }));
    } catch (error) {
      console.error('Failed to stop simulation:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addTerminalLine(`Failed to stop simulation: ${errorMessage}`, 'error');
      addSystemLog(`Failed to stop simulation: ${errorMessage}`, 'ERROR');
    }
  }, [addTerminalLine, addSystemLog, setStatus, buildUrl]);

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

  // Handle robot control commands
  const handleRobotCommand = useCallback(async (command: string) => {
    try {
      let response: Response;
      let data: any;

      if (command in MOVEMENT_PATH_MAP) {
        const direction = MOVEMENT_PATH_MAP[command] || command;
        response = await fetch(buildUrl(`/move/${direction}`), {
          method: 'POST',
        });
      } else {
        response = await fetch(buildUrl('/robot-command'), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ command }),
        });
      }

      try {
        data = await response.json();
      } catch {
        data = {};
      }
      
      if (!response.ok) {
        const errorMsg = data?.error || response.statusText || 'Unknown error';
        console.error(`Failed to send command "${command}":`, errorMsg);
        addTerminalLine(`Failed to send command "${command}": ${errorMsg}`, 'error');
        return;
      }
      
      console.log(`Command "${command}" sent successfully:`, data?.message || 'OK');
      addTerminalLine(`Robot command: ${command}`, 'info');
    } catch (error) {
      console.error('Failed to send robot command:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      addTerminalLine(`Failed to send robot command "${command}": ${errorMessage}`, 'error');
    }
  }, [addTerminalLine, buildUrl]); // Removed state.isRunning dependency

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
      // This logic was flawed, updating based on isRunning is tricky.
      // Let's just update regardless for the demo.
      updateMetric('cpu', Math.random() * 5);
      updateMetric('memory', Math.random() * 3 + 10);
    }, 1000);

    return () => {
      if (imagePollingIntervalRef.current) {
        clearInterval(imagePollingIntervalRef.current);
      }
      if (mockMetricsIntervalRef.current) {
        clearInterval(mockMetricsIntervalRef.current);
      }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only once on mount

  return (
    <div className="font-sans m-0 bg-gray-900 text-gray-200 flex h-screen overflow-hidden">
      <Sidebar
        onRunSimulation={handleRunSimulation}
        onStopSimulation={handleStopSimulation}
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
            <RobotControl onCommand={handleRobotCommand} isRunning={state.isRunning} />
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