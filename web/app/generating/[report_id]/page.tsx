"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { getReportStatus } from "../../../lib/api";

type Props = {
  params: { report_id: string };
};

export default function GeneratingPage({ params }: Props) {
  const reportId = useMemo(() => params.report_id, [params.report_id]);
  const router = useRouter();
  const [status, setStatus] = useState<"queued" | "processing" | "done" | "failed">("queued");
  const [error, setError] = useState("");
  const [dots, setDots] = useState(".");

  useEffect(() => {
    const dotTimer = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? "." : `${prev}.`));
    }, 480);

    return () => clearInterval(dotTimer);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const data = await getReportStatus(reportId);
        if (cancelled) return;

        setStatus(data.status);
        if (data.status === "done") {
          router.replace(`/report/${reportId}`);
          return;
        }
        if (data.status === "failed") {
          setError(data.error || "리포트 생성 중 문제가 발생했습니다.");
          return;
        }
      } catch (err) {
        if (cancelled) return;
        setStatus("failed");
        setError(err instanceof Error ? err.message : "상태 조회에 실패했습니다.");
        return;
      }

      if (!cancelled) timer = setTimeout(poll, 2000);
    };

    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [reportId, router]);

  return (
    <section className="section-space">
      <div className="mx-auto flex min-h-[45vh] w-full max-w-2xl flex-col items-center justify-center text-center">
        {status === "failed" ? (
          <>
            <h1 className="font-serif text-3xl tracking-tight">리포트 생성에 실패했습니다.</h1>
            <p className="mt-4 text-sm text-[#5f5f59]">{error}</p>
            <Link href="/start" className="mt-8 btn-primary">
              다시 시도하기
            </Link>
          </>
        ) : (
          <>
            <h1 className="font-serif text-3xl tracking-tight sm:text-4xl">리포트를 작성하고 있습니다.</h1>
            <p className="mt-4 text-base text-[#5f5f59]">약 30~60초 정도 소요됩니다{dots}</p>
            <p className="mt-6 text-xs text-[#8b8b84]">report_id: {reportId}</p>
          </>
        )}
      </div>
    </section>
  );
}
