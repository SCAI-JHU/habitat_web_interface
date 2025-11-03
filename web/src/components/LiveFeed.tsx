import React from 'react';

interface LiveFeedProps {
  imageSrc: string | null;
}

export const LiveFeed: React.FC<LiveFeedProps> = ({ imageSrc }) => {
  return (
    <div className="flex flex-col bg-gray-800 rounded-md border border-gray-700 flex-1">
      <h3 className="text-gray-300 text-base mb-0 text-center border-b border-gray-700 p-3 font-medium">
        Live Simulation Feed
      </h3>
      <div className="flex-1 flex flex-col items-center justify-center p-4 min-h-[300px]">
        {imageSrc ? (
          <img
            src={imageSrc}
            alt="Live simulation feed"
            className="border border-gray-600 max-w-full max-h-full h-auto"
          />
        ) : (
          <div className="text-gray-500 italic">
            No simulation feed available
          </div>
        )}
      </div>
    </div>
  );
};

