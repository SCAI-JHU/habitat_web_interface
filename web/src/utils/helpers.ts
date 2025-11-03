export const getCurrentTime = (): string => {
  return new Date().toTimeString().split(' ')[0];
};

export const generateId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

