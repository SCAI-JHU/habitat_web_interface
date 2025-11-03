import React from 'react';

interface SidebarProps {
  onRunSimulation: () => void;
  onRefreshImage: () => void;
  onClearTerminal: () => void;
  onClearLogs: () => void;
  isRunning: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  onRunSimulation,
  onRefreshImage,
  onClearTerminal,
  onClearLogs,
  isRunning
}) => {
  return (
    <aside className="w-72 bg-gray-950 flex-shrink-0 p-4 border-r border-gray-800 overflow-y-auto custom-scrollbar">
      <h2 className="text-xl font-semibold text-white mb-4">Configuration</h2>
      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">Simulation Controls</label>
          <div className="space-y-2">
            <button
              onClick={onRunSimulation}
              disabled={isRunning}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded-md font-medium hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              Run Simulation
            </button>
            <button
              onClick={onRefreshImage}
              className="w-full bg-gray-700 text-gray-200 border border-gray-600 py-2 px-4 rounded-md hover:bg-gray-600"
            >
              Refresh Image
            </button>
          </div>
        </div>

        <div>
          <label htmlFor="sim-model" className="block text-sm font-medium text-gray-400">
            Simulation Model
          </label>
          <select
            id="sim-model"
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded-md shadow-sm py-2 px-3 text-gray-200 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          >
            <option>HomeRobotSim (Default)</option>
            <option>Habitat 2.0</option>
            <option>iGibson</option>
          </select>
        </div>

        <div>
          <label htmlFor="scene-select" className="block text-sm font-medium text-gray-400">
            Scene
          </label>
          <select
            id="scene-select"
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded-md shadow-sm py-2 px-3 text-gray-200 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          >
            <option>Apartment (Default)</option>
            <option>Office</option>
            <option>Kitchen</option>
          </select>
        </div>

        <div>
          <label htmlFor="max-steps" className="block text-sm font-medium text-gray-400">
            Max Steps
          </label>
          <input
            type="number"
            id="max-steps"
            defaultValue="1000"
            className="mt-1 block w-full bg-gray-800 border border-gray-700 rounded-md shadow-sm py-2 px-3 text-gray-200 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">Terminal Controls</label>
          <button
            onClick={onClearTerminal}
            className="w-full bg-gray-700 text-gray-200 border border-gray-600 py-2 px-4 rounded-md hover:bg-gray-600"
          >
            Clear Terminal
          </button>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">Log Controls</label>
          <button
            onClick={onClearLogs}
            className="w-full bg-gray-700 text-gray-200 border border-gray-600 py-2 px-4 rounded-md hover:bg-gray-600"
          >
            Clear System Logs
          </button>
        </div>
      </div>
    </aside>
  );
};

