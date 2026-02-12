import "./globals.css";
import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "LoL 자동 해설 리포트",
  description: "Riot API + LLM MVP"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
