export type StatusType = 'error' | 'warning' | 'info' | 'success' | 'running' | 'complete' | 'idle';

export type MessageType = 'error' | 'warning' | 'info' | 'success';

export type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';

export interface TerminalLine {
  id: string;
  timestamp: string;
  message: string;
  type: MessageType;
}

export interface SystemLogLine {
  id: string;
  timestamp: string;
  message: string;
  level: LogLevel;
}

export interface MetricData {
  cpu: number[];
  memory: number[];
}

export interface AppState {
  status: string;
  statusType: StatusType;
  imageSrc: string | null;
  isRunning: boolean;
  terminalLines: TerminalLine[];
  systemLogs: SystemLogLine[];
  metrics: MetricData;
}

