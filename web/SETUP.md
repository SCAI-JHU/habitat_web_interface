# Setup Instructions

## Quick Start

1. **Install dependencies:**
   ```bash
   cd web
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```
   
   The app will be available at `http://localhost:5173`

3. **Build for production:**
   ```bash
   npm run build
   ```
   
   Output will be in `dist/` directory

## What's Different from the Old Version?

### Old (Vanilla JS)
- Single `index.html` file with inline JavaScript
- No build step
- Manual DOM manipulation
- No type safety

### New (React + TypeScript)
- Modern component-based architecture
- Type-safe with TypeScript
- Hot module replacement during development
- Better code organization and maintainability
- Easy to extend and test

## Project Structure

```
web/
├── src/
│   ├── components/       # React components
│   │   ├── Sidebar.tsx
│   │   ├── Header.tsx
│   │   ├── LiveFeed.tsx
│   │   ├── Metrics.tsx
│   │   ├── MetricsChart.tsx
│   │   ├── Terminal.tsx
│   │   └── SystemLogs.tsx
│   ├── hooks/           # Custom React hooks
│   │   └── useWebSocket.ts
│   ├── utils/           # Helper functions
│   │   └── helpers.ts
│   ├── types.ts         # TypeScript type definitions
│   ├── App.tsx          # Main application component
│   ├── main.tsx         # Application entry point
│   └── index.css        # Global styles
├── index.html           # HTML template
├── package.json         # Dependencies
├── tsconfig.json        # TypeScript configuration
├── vite.config.ts       # Vite configuration
├── tailwind.config.js   # Tailwind CSS configuration
└── postcss.config.js    # PostCSS configuration
```

## Development Tips

- **Hot Reload**: Changes to `.tsx` files will automatically reload in the browser
- **TypeScript**: Hover over variables in VSCode to see types
- **Linting**: Run `npm run lint` to check for code issues
- **Components**: Each component is self-contained and reusable

## Server Requirements

The React app expects the backend server to provide:

1. **HTTP Endpoints:**
   - `POST /run-simulation` - Starts the simulation
   - `GET /latest-image` - Returns latest image as JSON: `{image: "data:image/...", filename: "...", total_images: N}`

2. **WebSocket Endpoint:**
   - `WebSocket /ws/live_feed` - Real-time updates

3. **WebSocket Message Formats:**
   - Images: `data:image/png;base64,...`
   - Status: `status:type:message` (e.g., `status:running:Simulation started`)
   - Logs: `log:LEVEL:message` (e.g., `log:INFO:Processing frame 1`)
   - Metrics: `metric:type:value` (e.g., `metric:cpu:45.2`)

## Troubleshooting

**Port already in use:**
```bash
# Kill the process using port 5173
kill -9 $(lsof -ti:5173)
```

**Build fails:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**WebSocket not connecting:**
- Check that the backend server is running
- Verify the WebSocket URL in `src/hooks/useWebSocket.ts`
- Check browser console for connection errors

