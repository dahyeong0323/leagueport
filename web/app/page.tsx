import Link from "next/link";

export default function HomePage() {
  return (
    <section className="card">
      <h1>LoL 자동 해설 리포트</h1>
      <p className="muted">
        테스트 결제 후 Riot ID를 입력하면 최근 게임을 분석해 5섹션 리포트를 생성합니다.
      </p>
      <div className="row">
        <Link className="btn" href="/pay">
          시작하기
        </Link>
      </div>
    </section>
  );
}
