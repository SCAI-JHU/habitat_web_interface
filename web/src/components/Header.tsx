import React from 'react';
import { StatusType } from '../types';

interface HeaderProps {
  status: string;
  statusType: StatusType;
}

const statusColorMap: Record<StatusType, string> = {
  error: 'text-red-400 font-semibold',
  warning: 'text-yellow-400',
  info: 'text-gray-400',
  success: 'text-green-400',
  running: 'text-blue-400',
  complete: 'text-green-400',
  idle: 'text-gray-400'
};

export const Header: React.FC<HeaderProps> = ({ status, statusType }) => {
  const colorClass = statusColorMap[statusType] || 'text-gray-400';

  return (
    <header className="bg-gray-950 border-b border-gray-800 p-4 flex-shrink-0">
      <div className="flex justify-between items-center">
        <h1 className="text-xl text-gray-200 font-semibold">
          SCAI Lab Simulation Control Panel
        </h1>
        <div className={`italic text-center min-h-[22px] ${colorClass}`}>
          Status: {status}
        </div>
      </div>
    </header>
  );
};

