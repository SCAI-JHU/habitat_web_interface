import React from 'react';
import { MetricsChart } from './MetricsChart';
import { MetricData } from '../types';

interface MetricsProps {
  metrics: MetricData;
}

export const Metrics: React.FC<MetricsProps> = ({ metrics }) => {
  return (
    <div className="bg-gray-800 rounded-md border border-gray-700">
      <h3 className="text-gray-300 text-base mb-0 text-center border-b border-gray-700 p-3 font-medium">
        Simulation Metrics
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
        <MetricsChart
          data={metrics.cpu}
          label="CPU Usage (%)"
          borderColor="#3b82f6"
          backgroundColor="rgba(59, 130, 246, 0.3)"
        />
        <MetricsChart
          data={metrics.memory}
          label="Memory Usage (%)"
          borderColor="#10b981"
          backgroundColor="rgba(16, 185, 129, 0.3)"
        />
      </div>
    </div>
  );
};

