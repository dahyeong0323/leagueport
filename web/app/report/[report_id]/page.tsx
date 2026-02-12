"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getReport, getReportStatus, ReportResponse } from "../../../lib/api";

type Props = {
  params: { report_id: string };
};

export default function ReportPage({ params }: Props) {
  const reportId = useMemo(() => params.report_id, [params.report_id]);
  const [data, setData] = useState<ReportResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const status = await getReportStatus(reportId);
        if (status.status === "failed") {
          setError(status.error || "리포트 생성에 실패했습니다.");
          return;
        }
        if (status.status !== "done") {
          setError("아직 리포트가 완료되지 않았습니다.");
          return;
        }
        const report = await getReport(reportId);
        setData(report);
      } catch (err) {
        setError(err instanceof Error ? err.message : "리포트 조회에 실패했습니다.");
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, [reportId]);

  const onCopy = async () => {
    if (!data) return;
    const text = data.sections.map((section) => `## ${section.title}\n${section.content_markdown}`).join("\n\n");
    await navigator.clipboard.writeText(text);
  };

  if (loading) {
    return (
      <section className="card">
        <h1>리포트 불러오는 중...</h1>
      </section>
    );
  }

  if (error) {
    return (
      <section className="card">
        <h1>리포트 불러오기 실패</h1>
        <p style={{ color: "#b91c1c" }}>{error}</p>
        <div className="row">
          <Link className="btn secondary" href="/start">
            다시 시도
          </Link>
        </div>
      </section>
    );
  }

  if (!data) return null;

  return (
    <section className="card">
      <h1>LoL 자동 해설 리포트</h1>
      <p className="muted">
        {data.meta.riot_id} / {data.meta.region} / {data.meta.games_analyzed} games / {new Date(data.meta.created_at).toLocaleString()}
      </p>
      <div className="row" style={{ marginBottom: 12 }}>
        <button className="btn" onClick={onCopy}>
          전체 복사
        </button>
      </div>

      <h2>목차</h2>
      <ol>
        {data.sections.map((section) => (
          <li key={section.title}>{section.title}</li>
        ))}
      </ol>

      {data.sections.map((section) => (
        <article key={section.title} style={{ marginTop: 20 }}>
          <h3>{section.title}</h3>
          <ReactMarkdown>{section.content_markdown}</ReactMarkdown>
        </article>
      ))}
    </section>
  );
}
