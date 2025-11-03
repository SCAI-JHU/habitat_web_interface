import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface MetricsChartProps {
  data: number[];
  label: string;
  borderColor: string;
  backgroundColor: string;
}

export const MetricsChart: React.FC<MetricsChartProps> = ({
  data,
  label,
  borderColor,
  backgroundColor
}) => {
  const chartRef = useRef<ChartJS<'line'>>(null);

  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.update('none');
    }
  }, [data]);

  const chartData = {
    labels: Array(20).fill(''),
    datasets: [
      {
        label,
        data: data.slice(-20), // Keep last 20 data points
        borderColor,
        backgroundColor,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.3,
        fill: true
      }
    ]
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: { color: '#9ca3af' },
        grid: { color: '#374151' }
      },
      x: {
        ticks: { display: false },
        grid: { display: false }
      }
    },
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: { color: '#d1d5db' }
      },
      tooltip: {
        enabled: true
      }
    }
  };

  return (
    <div className="chart-container">
      <Line ref={chartRef} data={chartData} options={options} />
    </div>
  );
};

