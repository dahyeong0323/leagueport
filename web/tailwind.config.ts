import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./lib/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        accent: "#6f7d6d"
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Noto Sans KR", "sans-serif"],
        serif: ["Iowan Old Style", "Apple SD Gothic Neo", "Noto Serif KR", "serif"]
      }
    }
  },
  plugins: [require("@tailwindcss/typography")]
};

export default config;
