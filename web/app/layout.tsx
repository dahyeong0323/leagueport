import "./globals.css";
import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";

export const metadata: Metadata = {
  title: "롤 리포트",
  description: "Riot 전적 데이터를 기반으로 AI 리포트를 생성합니다."
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ko">
      <head>
        <meta charSet="utf-8" />
      </head>
      <body className="font-sans">
        <div className="mx-auto min-h-screen w-full max-w-5xl px-5 sm:px-8">
          <header className="flex items-center justify-between py-7">
            <Link href="/" className="font-serif text-lg tracking-tight">
              롤 리포트
            </Link>
            <Link href="/pay" className="text-sm text-[#3f3f3f] hover:text-[#111111]">
              리포트 만들기
            </Link>
          </header>

          <main>{children}</main>

          <footer className="border-t border-[#e5e5e2] py-6">
            <p className="text-xs text-[#6a6a66]">Powered by Riot API + GPT-5-mini. Riot Games is not affiliated.</p>
          </footer>
        </div>
      </body>
    </html>
  );
}
