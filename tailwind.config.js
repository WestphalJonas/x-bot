/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/web/templates/**/*.html',
    './src/web/static/js/**/*.js',
  ],
  safelist: [
    {
      pattern: /^(provider|operation)-/,
    },
    {
      pattern: /^log-status-(info|success|warning|error)$/,
    },
    {
      pattern: /^badge-/,
    },
    {
      pattern: /^mood-/,
    },
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
