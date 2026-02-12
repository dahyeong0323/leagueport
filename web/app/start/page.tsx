"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { createReport } from "../../lib/api";

export default function StartPage() {
  const router = useRouter();
  const [riotId, setRiotId] = useState("");
  const [region, setRegion] = useState("KR");
  const [tone, setTone] = useState<"funny" | "roast" | "sweet">("funny");
  const [language, setLanguage] = useState("ko");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const payload = {
        riot_id: riotId.trim(),
        region,
        tone,
        language
      };
      const result = await createReport(payload);
      router.push(`/generating/${result.report_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "리포트 생성 요청에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card">
      <h1>리포트 생성</h1>
      <form onSubmit={onSubmit}>
        <label htmlFor="riot-id">Riot ID</label>
        <input
          id="riot-id"
          placeholder="예: Hide on bush#KR1 또는 Hide on bush"
          value={riotId}
          onChange={(e) => setRiotId(e.target.value)}
          required
        />

        <label htmlFor="region">Region</label>
        <select id="region" value={region} onChange={(e) => setRegion(e.target.value)}>
          <option value="KR">KR</option>
          <option value="NA">NA</option>
          <option value="EUW">EUW</option>
        </select>

        <label htmlFor="tone">Tone</label>
        <select id="tone" value={tone} onChange={(e) => setTone(e.target.value as "funny" | "roast" | "sweet")}>
          <option value="funny">funny</option>
          <option value="roast">roast</option>
          <option value="sweet">sweet</option>
        </select>

        <label htmlFor="language">Language</label>
        <select id="language" value={language} onChange={(e) => setLanguage(e.target.value)}>
          <option value="ko">ko</option>
          <option value="en">en</option>
        </select>

        {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}

        <button className="btn" type="submit" disabled={loading}>
          {loading ? "요청 중..." : "리포트 만들기"}
        </button>
      </form>
    </section>
  );
}
