"use client";

import { useRouter } from "next/navigation";

export default function PayPage() {
  const router = useRouter();

  return (
    <section className="card">
      <h1>테스트 결제</h1>
      <p className="muted">MVP 단계에서는 실제 결제 대신 테스트 버튼만 제공합니다.</p>
      <button className="btn" onClick={() => router.push("/start")}>
        결제 완료(테스트)
      </button>
    </section>
  );
}
