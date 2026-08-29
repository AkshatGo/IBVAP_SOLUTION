/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'ibvap': {
          'dark': '#0d1117',
          'darker': '#161b22',
          'card': '#1e1e2e',
          'border': '#30363d',
          'text': '#ffffff',
          'muted': '#8b949e',
          'accent': '#58a6ff',
          'success': '#4caf50',
          'warning': '#ff9800',
          'danger': '#f44336',
          'critical': '#9c27b0',
        }
      },
      fontFamily: {
        'sans': ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        'mono': ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
