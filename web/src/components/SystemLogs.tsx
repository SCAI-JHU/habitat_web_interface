import React, { useEffect, useRef } from 'react';
import { SystemLogLine, LogLevel } from '../types';

interface SystemLogsProps {
  logs: SystemLogLine[];
}

const logLevelColorMap: Record<LogLevel, string> = {
  INFO: 'text-blue-400',
  WARN: 'text-yellow-400',
  ERROR: 'text-red-400',
  DEBUG: 'text-purple-400'
};

export const SystemLogs: React.FC<SystemLogsProps> = ({ logs }) => {
  const logsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="flex flex-col bg-gray-800 rounded-md border border-gray-700 h-80">
      <h3 className="text-gray-300 text-base mb-0 text-center border-b border-gray-700 p-3 font-medium">
        System Logs
      </h3>
      <div
        ref={logsRef}
        className="flex-1 bg-black p-3 font-mono text-xs leading-relaxed overflow-y-auto custom-scrollbar"
      >
        {logs.map((log) => (
          <div key={log.id} className="log-line">
            <span className="text-gray-500 mr-2">[{log.timestamp}]</span>
            <span className={`font-medium ${logLevelColorMap[log.level] || 'text-blue-400'}`}>
              [{log.level}]
            </span>
            <span className="text-gray-300 ml-1">{log.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

