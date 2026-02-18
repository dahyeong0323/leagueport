"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getReport, getReportStatus, ReportResponse } from "../../../lib/api";

type Props = {
  params: { report_id: string };
};

type ReportSection = {
  id: string;
  title: string;
  markdown: string;
  isOverview?: boolean;
};

const sectionMoodOverlays = [
  "radial-gradient(circle at 50% 18%, rgba(71, 85, 105, 0.24), transparent 58%), linear-gradient(180deg, rgba(37, 45, 58, 0.28), rgba(15, 15, 20, 0) 70%)",
  "radial-gradient(circle at 50% 18%, rgba(99, 102, 241, 0.22), transparent 58%), linear-gradient(180deg, rgba(47, 46, 89, 0.28), rgba(15, 15, 20, 0) 70%)",
  "radial-gradient(circle at 50% 18%, rgba(225, 29, 72, 0.18), transparent 58%), linear-gradient(180deg, rgba(84, 34, 59, 0.28), rgba(15, 15, 20, 0) 70%)",
  "radial-gradient(circle at 50% 18%, rgba(217, 119, 6, 0.2), transparent 58%), linear-gradient(180deg, rgba(88, 54, 34, 0.28), rgba(15, 15, 20, 0) 70%)",
  "radial-gradient(circle at 50% 18%, rgba(5, 150, 105, 0.2), transparent 58%), linear-gradient(180deg, rgba(28, 72, 59, 0.28), rgba(15, 15, 20, 0) 70%)",
  "radial-gradient(circle at 50% 18%, rgba(2, 132, 199, 0.2), transparent 58%), linear-gradient(180deg, rgba(25, 58, 80, 0.28), rgba(15, 15, 20, 0) 70%)"
];

function displayNameFromRiotId(riotId: string) {
  const [name] = riotId.split("#");
  return name || riotId;
}

function splitTitleForRhythm(title: string) {
  const words = title.trim().split(/\s+/).filter(Boolean);
  if (words.length < 2) return { line1: title, line2: "" };

  const splitIndex = Math.ceil(words.length / 2);
  return {
    line1: words.slice(0, splitIndex).join(" "),
    line2: words.slice(splitIndex).join(" ")
  };
}

async function copyText(text: string): Promise<boolean> {
  if (typeof window === "undefined") return false;

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fallback below
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    document.body.removeChild(textarea);
    return copied;
  } catch {
    return false;
  }
}

export default function ReportPage({ params }: Props) {
  const router = useRouter();
  const reportId = useMemo(() => params.report_id, [params.report_id]);
  const [data, setData] = useState<ReportResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [activeIndex, setActiveIndex] = useState(0);
  const [expandedSections, setExpandedSections] = useState<Record<number, boolean>>({});
  const mainRef = useRef<HTMLElement | null>(null);
  const sectionRefs = useRef<Array<HTMLElement | null>>([]);
  const activeIndexRef = useRef(0);
  const [bgFromIndex, setBgFromIndex] = useState(0);
  const [bgToIndex, setBgToIndex] = useState(0);
  const [bgOverlayVisible, setBgOverlayVisible] = useState(false);
  const bgFadeTimeoutRef = useRef<number | null>(null);
  const bgFadeRafRef = useRef<number | null>(null);
  const currentBgIndexRef = useRef(0);
  const [copiedState, setCopiedState] = useState<"report" | "invite" | null>(null);
  const [copyError, setCopyError] = useState("");
  const copiedResetRef = useRef<number | null>(null);

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
          setError("리포트가 아직 완성되지 않았습니다.");
          return;
        }

        const report = await getReport(reportId);
        setData(report);
      } catch (err) {
        setError(err instanceof Error ? err.message : "리포트를 불러오지 못했습니다.");
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, [reportId]);

  const reportSections = useMemo<ReportSection[]>(() => {
    if (!data) return [];

    const displayName = displayNameFromRiotId(data.meta.riot_id);
    const overviewMarkdown = [
      `## ${displayName}님의 최근 ${data.meta.games_analyzed}판 리포트`,
      `${data.meta.region} 서버 기준 · ${new Date(data.meta.created_at).toLocaleString("ko-KR")}`,
      data.meta.data_source === "fallback"
        ? "일시적으로 Riot 실시간 데이터를 불러오지 못해 대체 데이터 기준으로 생성했습니다."
        : "Riot 공식 API 기반 데이터로 생성했습니다."
    ].join("\n\n");

    return [
      {
        id: "overview",
        title: "리포트 개요",
        markdown: overviewMarkdown,
        isOverview: true
      },
      ...data.sections.map((section, index) => ({
        id: `section-${index}`,
        title: section.title,
        markdown: section.content_markdown
      }))
    ];
  }, [data]);

  useEffect(() => {
    activeIndexRef.current = activeIndex;
  }, [activeIndex]);

  useEffect(() => {
    const nextIndex = activeIndex % sectionMoodOverlays.length;
    if (nextIndex === currentBgIndexRef.current) return;

    if (bgFadeTimeoutRef.current) {
      window.clearTimeout(bgFadeTimeoutRef.current);
      bgFadeTimeoutRef.current = null;
    }
    if (bgFadeRafRef.current) {
      window.cancelAnimationFrame(bgFadeRafRef.current);
      bgFadeRafRef.current = null;
    }

    setBgFromIndex(currentBgIndexRef.current);
    setBgToIndex(nextIndex);
    setBgOverlayVisible(false);

    bgFadeRafRef.current = window.requestAnimationFrame(() => {
      setBgOverlayVisible(true);
    });

    bgFadeTimeoutRef.current = window.setTimeout(() => {
      currentBgIndexRef.current = nextIndex;
      setBgFromIndex(nextIndex);
      setBgOverlayVisible(false);
    }, 650);
  }, [activeIndex]);

  useEffect(
    () => () => {
      if (bgFadeTimeoutRef.current) window.clearTimeout(bgFadeTimeoutRef.current);
      if (bgFadeRafRef.current) window.cancelAnimationFrame(bgFadeRafRef.current);
      if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current);
    },
    []
  );

  useEffect(() => {
    const mainElement = mainRef.current;
    if (!mainElement || reportSections.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        let nextIndex = activeIndexRef.current;
        let maxRatio = 0;

        entries.forEach((entry) => {
          const sectionIndex = Number((entry.target as HTMLElement).dataset.sectionIndex);
          if (!Number.isNaN(sectionIndex) && entry.intersectionRatio > maxRatio) {
            maxRatio = entry.intersectionRatio;
            nextIndex = sectionIndex;
          }
        });

        if (nextIndex !== activeIndexRef.current) {
          setActiveIndex(nextIndex);
        }
      },
      {
        root: mainElement,
        threshold: [0.35, 0.6, 0.8]
      }
    );

    sectionRefs.current.forEach((section) => {
      if (section) observer.observe(section);
    });

    return () => observer.disconnect();
  }, [reportSections.length]);

  const scrollToSection = (index: number) => {
    sectionRefs.current[index]?.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });
  };

  const toggleExpanded = (index: number) => {
    setExpandedSections((prev) => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const goHome = () => {
    try {
      router.push("/start");
    } catch {
      window.location.assign("/start");
    }
  };

  const onCopy = async (target: "report" | "invite") => {
    const text = target === "report" ? window.location.href : "http://localhost:3000";
    const copied = await copyText(text);
    if (!copied) {
      setCopyError("복사에 실패했습니다.");
      return;
    }

    setCopyError("");
    setCopiedState(target);
    if (copiedResetRef.current) window.clearTimeout(copiedResetRef.current);
    copiedResetRef.current = window.setTimeout(() => {
      setCopiedState((prev) => (prev === target ? null : prev));
    }, 1500);
  };

  const isLongSection = (markdown: string) => markdown.length > 460;

  if (loading) {
    return (
      <main className="relative flex h-screen items-center justify-center overflow-hidden bg-[#0f0f14] px-6 text-[#ececf4]">
        <div aria-hidden className="pointer-events-none absolute inset-0 report-radial-glow" />
        <div className="w-full max-w-xl rounded-[24px] border border-white/[0.08] bg-white/[0.04] p-10 text-center shadow-[0_0_40px_rgba(120,0,255,0.08)] backdrop-blur-[20px]">
          <p className="text-sm tracking-[0.08em] text-[#b2b6cc]">롤포트 · Leagueport</p>
          <p className="mt-5 text-lg text-[#ececf4]">리포트를 불러오고 있습니다.</p>
        </div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="relative flex h-screen items-center justify-center overflow-hidden bg-[#0f0f14] px-6 text-[#ececf4]">
        <div aria-hidden className="pointer-events-none absolute inset-0 report-radial-glow" />
        <div className="w-full max-w-xl rounded-[24px] border border-white/[0.08] bg-white/[0.04] p-10 text-center shadow-[0_0_40px_rgba(120,0,255,0.08)] backdrop-blur-[20px]">
          <p className="text-sm tracking-[0.08em] text-[#b2b6cc]">롤포트 · Leagueport</p>
          <p className="mt-5 text-base text-[#ffafc6]">{error}</p>
          <Link href="/start" className="mt-7 inline-flex btn-primary">
            새로 생성하기
          </Link>
        </div>
      </main>
    );
  }

  if (!data) return null;

  const shareSectionIndex = reportSections.length;
  const navSectionCount = reportSections.length + 1;

  return (
    <div className="relative isolate h-screen overflow-hidden bg-[#0f0f14] text-[#ececf4]">
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-20 bg-[#0f0f14]" />
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 report-radial-glow" />
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-10">
        <div className="absolute inset-0 report-bg-fade" style={{ backgroundImage: sectionMoodOverlays[bgFromIndex] }} />
        <div
          className={`absolute inset-0 report-bg-fade ${bgOverlayVisible ? "opacity-100" : "opacity-0"}`}
          style={{ backgroundImage: sectionMoodOverlays[bgToIndex] }}
        />
      </div>
      <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 report-noise" />

      <aside className="fixed left-6 top-6 z-30 rounded-[20px] border border-white/[0.08] bg-white/[0.03] px-5 py-4 shadow-[0_0_30px_rgba(120,0,255,0.08)] backdrop-blur-[16px]">
        <p className="font-serif text-xl leading-none tracking-tight text-[#f3f3fb]">롤포트</p>
        <p className="mt-1 text-xs tracking-[0.14em] text-[#a8aec8]">Leagueport</p>
        <button
          type="button"
          onClick={goHome}
          className="mt-5 inline-flex text-sm text-[#d2d6ea] underline underline-offset-4"
        >
          ← 메인으로 돌아가기
        </button>
      </aside>

      <nav aria-label="리포트 섹션 이동" className="fixed right-5 top-1/2 z-30 -translate-y-1/2">
        <ul className="flex flex-col gap-3">
          {Array.from({ length: navSectionCount }).map((_, index) => {
            const isActive = index === activeIndex;
            const label = index === shareSectionIndex ? "친구에게 공유" : `섹션 ${index + 1}`;
            return (
              <li key={label}>
                <button
                  type="button"
                  aria-label={`섹션 ${index + 1}로 이동`}
                  aria-current={isActive ? "true" : undefined}
                  onClick={() => scrollToSection(index)}
                  className={`h-3.5 w-3.5 rounded-full border transition-all duration-300 ${
                    isActive
                      ? "scale-125 border-[#8f72ff] bg-[#8f72ff]"
                      : "border-[#636984] bg-white/[0.12] hover:border-[#8f72ff]"
                  } focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#a78bfa] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0f0f14]`}
                />
              </li>
            );
          })}
        </ul>
      </nav>

      <main ref={mainRef} className="relative z-10 h-screen overflow-y-auto snap-y snap-mandatory scroll-smooth no-scrollbar">
        {reportSections.map((section, index) => {
          const isActive = index === activeIndex;
          const isExpanded = !!expandedSections[index];
          const shouldCollapse = !section.isOverview && isLongSection(section.markdown);
          const isReadMode = shouldCollapse && isExpanded;
          const { line1, line2 } = splitTitleForRhythm(section.title);

          return (
            <section
              key={section.id}
              data-section-index={index}
              ref={(element) => {
                sectionRefs.current[index] = element;
              }}
              className={`min-h-screen snap-start snap-always flex justify-center px-6 ${
                isReadMode ? "items-start py-16 sm:py-20" : "items-center py-8 sm:py-12"
              }`}
            >
              <div
                className={`section-card-transition max-w-3xl w-full rounded-[24px] border border-white/[0.08] bg-white/[0.04] p-8 shadow-[0_0_40px_rgba(120,0,255,0.08)] backdrop-blur-[20px] sm:p-12 ${
                  isActive ? "translate-y-0 opacity-100" : "translate-y-5 opacity-70"
                }`}
              >
                <p className="section-label">
                  Section {String(index + 1).padStart(2, "0")}
                </p>
                <h2 className="section-title mt-4">
                  <span className="section-title-line">{line1}</span>
                  {line2 ? <span className="section-title-emphasis highlight">{line2}</span> : null}
                </h2>

                <article
                  className={`prose prose-invert mt-8 max-w-none text-base leading-7 prose-headings:font-serif prose-headings:text-[#f2f3fb] prose-p:text-[#d6d9e7] prose-strong:text-[#f5f6ff] prose-li:text-[#d6d9e7] prose-a:text-[#c4b5fd] sm:text-lg sm:leading-8 ${
                    shouldCollapse && !isExpanded ? "max-h-[38vh] overflow-hidden" : ""
                  }`}
                >
                  <ReactMarkdown>{section.markdown}</ReactMarkdown>
                </article>

                {shouldCollapse ? (
                  <div className="mt-6">
                    <button
                      type="button"
                      onClick={() => toggleExpanded(index)}
                      className="rounded-full border border-white/[0.16] bg-white/[0.04] px-4 py-2 text-sm text-[#d7dbef] transition-colors duration-300 hover:bg-white/[0.08] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#a78bfa] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0f0f14]"
                    >
                      {isExpanded ? "접기" : "더 보기"}
                    </button>
                  </div>
                ) : null}
              </div>
            </section>
          );
        })}

        <section
          data-section-index={shareSectionIndex}
          ref={(element) => {
            sectionRefs.current[shareSectionIndex] = element;
          }}
          className="min-h-screen snap-start snap-always flex items-center justify-center px-6 py-8 sm:py-12"
        >
          <div
            className={`section-card-transition max-w-3xl w-full rounded-[24px] border border-white/[0.08] bg-white/[0.04] p-8 shadow-[0_0_40px_rgba(120,0,255,0.08)] backdrop-blur-[20px] sm:p-12 ${
              activeIndex === shareSectionIndex ? "translate-y-0 opacity-100" : "translate-y-5 opacity-70"
            }`}
          >
            <p className="section-label">
              Section {String(shareSectionIndex + 1).padStart(2, "0")}
            </p>
            <h2 className="section-title mt-4">
              <span className="section-title-line">친구에게 공유</span>
            </h2>
            <div className="mt-8 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => onCopy("report")}
                className="rounded-full border border-white/[0.16] bg-white/[0.04] px-4 py-2 text-sm text-[#d7dbef] transition-colors duration-300 hover:bg-white/[0.08] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#a78bfa] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0f0f14]"
              >
                {copiedState === "report" ? "복사됨" : "리포트 링크 복사"}
              </button>
              <button
                type="button"
                onClick={() => onCopy("invite")}
                className="rounded-full border border-white/[0.16] bg-white/[0.04] px-4 py-2 text-sm text-[#d7dbef] transition-colors duration-300 hover:bg-white/[0.08] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#a78bfa] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0f0f14]"
              >
                {copiedState === "invite" ? "복사됨" : "친구초대 링크 복사"}
              </button>
            </div>
            {copyError ? <p className="mt-4 text-sm text-[#ffafc6]">{copyError}</p> : null}
          </div>
        </section>
      </main>
    </div>
  );
}
