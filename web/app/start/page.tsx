"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { createReport } from "../../lib/api";

const regions = ["EUW", "KR"] as const;
type RegionOption = (typeof regions)[number];

export default function StartPage() {
  const router = useRouter();
  const [riotId, setRiotId] = useState("");
  const [region, setRegion] = useState<RegionOption>("EUW");
  const [language, setLanguage] = useState("ko");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async (event: FormEvent) => {
    event.preventDefault();

    const trimmedRiotId = riotId.trim();
    if (!trimmedRiotId.includes("#")) {
      setError("Riot ID 형식은 GameName#TAG 입니다.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      if (!regions.includes(region)) {
        throw new Error("지원하지 않는 서버입니다. EUW 또는 KR만 선택 가능합니다.");
      }
      const result = await createReport({
        riot_id: trimmedRiotId,
        region,
        tone: "funny",
        language
      });
      router.push(`/generating/${result.report_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "리포트 생성 요청에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="section-space">
      <div className="mx-auto w-full max-w-[480px]">
        <h1 className="font-serif text-3xl tracking-tight">리포트 만들기</h1>
        <p className="mt-3 text-sm text-[#6a6a66]">Riot ID를 입력하면 최근 20판 기준으로 리포트를 생성합니다.</p>

        <form className="mt-10 space-y-6" onSubmit={onSubmit}>
          <div>
            <label htmlFor="riot-id" className="mb-2 block text-sm">
              Riot ID
            </label>
            <input
              id="riot-id"
              value={riotId}
              onChange={(e) => setRiotId(e.target.value)}
              className="input-minimal"
              placeholder="GameName#TAG"
              required
            />
            <p className="mt-2 text-xs text-[#767671]">예: Hide on bush#KR1</p>
          </div>

          <div>
            <label htmlFor="region" className="mb-2 block text-sm">
              Region
            </label>
            <select
              id="region"
              value={region}
              onChange={(e) => {
                const value = e.target.value as RegionOption;
                if (regions.includes(value)) setRegion(value);
              }}
              className="input-minimal"
            >
              {regions.map((regionOption) => (
                <option key={regionOption} value={regionOption}>
                  {regionOption}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label htmlFor="language" className="mb-2 block text-sm">
              언어 선택
            </label>
            <select
              id="language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="input-minimal"
            >
              <option value="ko">한국어</option>
              <option value="en">English</option>
            </select>
          </div>

          {error ? <p className="text-sm text-[#a64d3d]">{error}</p> : null}

          <button type="submit" disabled={loading} className="btn-primary w-full">
            {loading ? "생성 중..." : "1,000원 결제 후 리포트 생성"}
          </button>
        </form>
      </div>
    </section>
  );
}
