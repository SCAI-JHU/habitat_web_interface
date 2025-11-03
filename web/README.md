# SCAI Lab Simulation Control Panel - React TypeScript

This is a modern React TypeScript implementation of the simulation control panel.

## Features

- Real-time WebSocket communication for live updates
- Interactive charts for CPU and memory metrics (Chart.js)
- Live simulation feed display
- Terminal console and system logs
- Dark theme with Tailwind CSS

## Development

### Prerequisites

- Node.js 16+ and npm/yarn

### Installation

```bash
npm install
```

### Development Server

```bash
npm run dev
```

This will start the Vite development server at `http://localhost:5173` with hot module replacement.

### Build for Production

```bash
npm run build
```

The production-ready files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Architecture

### Components

- `App.tsx` - Main application component with state management
- `Sidebar.tsx` - Configuration panel and controls
- `Header.tsx` - Status bar and title
- `LiveFeed.tsx` - Real-time simulation image display
- `Metrics.tsx` - CPU and memory usage charts
- `Terminal.tsx` - Command output console
- `SystemLogs.tsx` - System log viewer
- `MetricsChart.tsx` - Reusable chart component

### Hooks

- `useWebSocket.ts` - Custom hook for WebSocket connection management

### Types

- `types.ts` - TypeScript interfaces and type definitions

## Server Integration

The app expects the following endpoints:

- `POST /run-simulation` - Start simulation
- `GET /latest-image` - Fetch latest simulation image
- `WebSocket /ws/live_feed` - Real-time updates

WebSocket message formats:
- `data:image/...` - Base64 encoded image
- `status:type:message` - Status updates
- `log:LEVEL:message` - System logs
- `metric:type:value` - Metrics (cpu/mem)

## Migration from Vanilla JS

The original vanilla JavaScript implementation is preserved as `index_old.html` for reference. This React TypeScript version provides:

- Better code organization and maintainability
- Type safety with TypeScript
- Component reusability
- Modern development tooling with Vite
- Hot module replacement for faster development

