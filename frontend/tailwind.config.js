/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Plus Jakarta Sans', 'system-ui', 'sans-serif'],
        body: ['Inter', 'system-ui', 'sans-serif'],
        serif: ['Merriweather', 'Georgia', 'serif'],
      },
      colors: {
        primary: {
          DEFAULT: '#4338ca',
          foreground: '#ffffff',
          hover: '#3730a3',
        },
        secondary: {
          DEFAULT: '#f1f5f9',
          foreground: '#0f172a',
        },
        accent: {
          DEFAULT: '#e0e7ff',
          foreground: '#4338ca',
        },
        background: {
          light: '#ffffff',
          dark: '#020617',
          paper: '#fafaf9',
        },
        surface: {
          light: '#ffffff',
          dark: '#0f172a',
        },
      },
      borderRadius: {
        lg: '0.5rem',
        md: '0.375rem',
        sm: '0.25rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-in-out',
        'slide-up': 'slideUp 0.4s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}