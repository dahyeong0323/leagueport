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
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<"queued" | "processing" | "done" | "failed">("queued");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const data = await getReportStatus(reportId);
        if (cancelled) return;
        setProgress(data.progress);
        setStatus(data.status);
        if (data.status === "done") {
          router.replace(`/report/${reportId}`);
          return;
        }
        if (data.status === "failed") {
          setError(data.error || "리포트 생성에 실패했습니다.");
          return;
        }
      } catch (err) {
        if (!cancelled) {
          setStatus("failed");
          setError(err instanceof Error ? err.message : "상태 조회에 실패했습니다.");
        }
      }
      if (!cancelled) {
        setTimeout(poll, 2000);
      }
    };

    void poll();
    return () => {
      cancelled = true;
    };
  }, [reportId, router]);

  return (
    <section className="card">
      <h1>리포트 생성 중</h1>
      <p className="muted">report_id: {reportId}</p>
      <div className="progress" style={{ margin: "14px 0" }}>
        <div style={{ width: `${progress}%`, transition: "width .4s ease" }} />
      </div>
      <p>
        상태: <b>{status}</b> ({progress}%)
      </p>

      {status === "failed" ? (
        <>
          <p style={{ color: "#b91c1c" }}>실패 이유: {error}</p>
          <div className="row">
            <Link className="btn secondary" href="/start">
              다시 시도
            </Link>
          </div>
        </>
      ) : null}
    </section>
  );
}
