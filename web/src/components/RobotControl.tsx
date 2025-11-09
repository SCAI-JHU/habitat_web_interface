import { useState, useEffect, useCallback } from 'react';

interface RobotControlProps {
  onCommand: (command: string) => void;
  isRunning: boolean;
}

type ControlKey = 'forward' | 'backward' | 'left' | 'right' | 'arm_up' | 'arm_down' | 'grip_open' | 'grip_close' | 'stop';

const CONTROL_MAPPING: Record<string, ControlKey> = {
  'w': 'forward',
  's': 'backward',
  'a': 'left',
  'd': 'right',
  'e': 'arm_up',
  'q': 'arm_down',
  'o': 'grip_open',
  'p': 'grip_close',
};

const KEY_LABELS: Record<ControlKey, string> = {
  forward: 'W - Forward',
  backward: 'S - Back',
  left: 'A - Left',
  right: 'D - Right',
  arm_up: 'E - Arm Up',
  arm_down: 'Q - Arm Down',
  grip_open: 'O - Grip Open',
  grip_close: 'P - Grip Close',
  stop: 'Stop',
};

export const RobotControl = ({ onCommand, isRunning }: RobotControlProps) => {
  const [pressedKeys, setPressedKeys] = useState<Set<string>>(new Set());
  const [isKeyboardMode, setIsKeyboardMode] = useState(false); // Default to buttons

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!isRunning || !isKeyboardMode) return;
    
    const key = event.key.toLowerCase();
    const control = CONTROL_MAPPING[key];
    
    if (control && !pressedKeys.has(key)) {
      setPressedKeys(prev => new Set(prev).add(key));
      onCommand(control);
    }
  }, [isRunning, isKeyboardMode, pressedKeys, onCommand]);

  const handleKeyUp = useCallback((event: KeyboardEvent) => {
    const key = event.key.toLowerCase();
    if (pressedKeys.has(key)) {
      setPressedKeys(prev => {
        const newSet = new Set(prev);
        newSet.delete(key);
        return newSet;
      });
      onCommand('stop');
    }
  }, [pressedKeys, onCommand]);

  useEffect(() => {
    if (isKeyboardMode && isRunning) {
      window.addEventListener('keydown', handleKeyDown);
      window.addEventListener('keyup', handleKeyUp);
      
      return () => {
        window.removeEventListener('keydown', handleKeyDown);
        window.removeEventListener('keyup', handleKeyUp);
      };
    }
  }, [isKeyboardMode, isRunning, handleKeyDown, handleKeyUp]);

  const handleButtonClick = (command: ControlKey) => {
    console.log(`Robot control button: ${command} (sim running: ${isRunning})`);
    onCommand(command);

    // For movement commands during an active sim, send a short stop impulse
    if (
      isRunning &&
      command !== 'stop' &&
      (command === 'forward' || command === 'backward' || command === 'left' || command === 'right')
    ) {
      setTimeout(() => {
        console.log(`Auto-stopping after ${command}`);
        onCommand('stop');
      }, 250); // 250ms delay for discrete movement
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-200">Robot Control</h3>
        <button
          onClick={() => setIsKeyboardMode(!isKeyboardMode)}
          className={`px-3 py-1 text-sm rounded ${
            isKeyboardMode 
              ? 'bg-blue-600 hover:bg-blue-700' 
              : 'bg-gray-600 hover:bg-gray-700'
          } text-white`}
          title="Toggle between button and keyboard controls"
        >
          {isKeyboardMode ? '⌨️ Keyboard' : '🔘 Buttons'}
        </button>
      </div>

      {!isRunning && (
        <div className="text-yellow-400 text-sm mb-4">
          Simulation not marked as running. Commands will still be sent, but the backend may reject them until the sim is ready.
        </div>
      )}

      {isKeyboardMode ? (
        <div className="space-y-2">
          <div className="text-sm text-gray-400 mb-3">Press keys to control robot:</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {Object.entries(KEY_LABELS).map(([key, label]) => (
              <div
                key={key}
                className={`p-2 rounded border ${
                  pressedKeys.has(key.charAt(0).toLowerCase())
                    ? 'bg-blue-600 border-blue-500'
                    : 'bg-gray-700 border-gray-600'
                }`}
              >
                {label}
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="text-sm text-gray-400 mb-2">Click buttons to control:</div>
          
          {/* Movement Controls */}
          <div className="space-y-2">
            <div className="text-xs text-gray-500 mb-1">Movement</div>
            <div className="grid grid-cols-3 gap-2">
              <div></div>
              <button
                onClick={() => handleButtonClick('forward')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  isRunning ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-blue-900 text-blue-200'
                }`}
              >
                ↑ Forward
              </button>
              <div></div>
              <button
                onClick={() => handleButtonClick('left')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  isRunning ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-blue-900 text-blue-200'
                }`}
              >
                ← Left
              </button>
              <button
                onClick={() => handleButtonClick('stop')}
                className="px-3 py-2 bg-red-700 hover:bg-red-800 text-white rounded text-sm font-bold transition-colors"
              >
                ⏹ Stop
              </button>
              <button
                onClick={() => handleButtonClick('right')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  isRunning ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-blue-900 text-blue-200'
                }`}
              >
                → Right
              </button>
              <div></div>
              <button
                onClick={() => handleButtonClick('backward')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  isRunning ? 'bg-blue-600 hover:bg-blue-700 text-white' : 'bg-blue-900 text-blue-200'
                }`}
              >
                ↓ Back
              </button>
              <div></div>
            </div>
          </div>

          {/* Arm Controls */}
          <div className="space-y-2">
            <div className="text-xs text-gray-500 mb-1">Arm</div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => handleButtonClick('arm_up')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  isRunning ? 'bg-green-600 hover:bg-green-700 text-white' : 'bg-green-900 text-green-200'
                }`}
              >
                ↑ Arm Up
              </button>
              <button
                onClick={() => handleButtonClick('arm_down')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  isRunning ? 'bg-green-600 hover:bg-green-700 text-white' : 'bg-green-900 text-green-200'
                }`}
              >
                ↓ Arm Down
              </button>
            </div>
          </div>

          {/* Gripper Controls */}
          <div className="space-y-2">
            <div className="text-xs text-gray-500 mb-1">Gripper</div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => handleButtonClick('grip_open')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  isRunning ? 'bg-purple-600 hover:bg-purple-700 text-white' : 'bg-purple-900 text-purple-200'
                }`}
              >
                ✋ Open
              </button>
              <button
                onClick={() => handleButtonClick('grip_close')}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  isRunning ? 'bg-purple-600 hover:bg-purple-700 text-white' : 'bg-purple-900 text-purple-200'
                }`}
              >
                ✊ Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

