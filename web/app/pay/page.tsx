"use client";

import { useRouter } from "next/navigation";

export default function PayPage() {
  const router = useRouter();

  return (
    <section className="section-space">
      <div className="mx-auto w-full max-w-[480px]">
        <h1 className="font-serif text-3xl tracking-tight">테스트 결제</h1>
        <p className="mt-4 text-sm leading-relaxed text-[#5f5f59]">
          MVP 단계에서는 실제 결제 대신 테스트 결제 버튼을 제공합니다. 결제를 완료하면 Riot ID 입력 화면으로
          이동합니다.
        </p>

        <div className="mt-12">
          <button className="btn-primary w-full" onClick={() => router.push("/start")}>
            결제 완료하기
          </button>
        </div>
      </div>
    </section>
  );
}
