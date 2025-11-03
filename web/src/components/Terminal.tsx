import React, { useEffect, useRef } from 'react';
import { TerminalLine, MessageType } from '../types';

interface TerminalProps {
  lines: TerminalLine[];
}

const terminalColorMap: Record<MessageType, string> = {
  error: 'text-red-400',
  warning: 'text-yellow-400',
  info: 'text-gray-200',
  success: 'text-green-400'
};

export const Terminal: React.FC<TerminalProps> = ({ lines }) => {
  const terminalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [lines]);

  return (
    <div className="flex flex-col bg-gray-800 rounded-md border border-gray-700 h-80">
      <h3 className="text-gray-300 text-base mb-0 text-center border-b border-gray-700 p-3 font-medium">
        Terminal Console
      </h3>
      <div
        ref={terminalRef}
        className="flex-1 bg-black p-3 font-mono text-xs leading-relaxed overflow-y-auto custom-scrollbar"
      >
        {lines.map((line) => (
          <div key={line.id} className="terminal-line">
            <span className="text-gray-500 mr-2">[{line.timestamp}]</span>
            <span className="text-gray-400 mr-1">$</span>
            <span className={terminalColorMap[line.type] || 'text-gray-200'}>
              {line.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

