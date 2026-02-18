import Link from "next/link";

export default function HomePage() {
  return (
    <div className="pb-12">
      <section className="section-space text-center">
        <h1 className="mx-auto max-w-3xl whitespace-pre-line font-serif text-3xl leading-tight tracking-tight sm:text-5xl">
          {"최근 20판,\n당신의 플레이는 어떤 흐름을 그리고 있을까."}
        </h1>
        <p className="mx-auto mt-6 max-w-xl whitespace-pre-line text-sm leading-relaxed text-[#5f5f59] sm:text-base">
          {"Riot 전적 데이터를 기반으로\nAI가 당신의 플레이 패턴을 정리합니다."}
        </p>
        <div className="mt-10">
          <Link href="/pay" className="btn-primary">
            리포트 만들기
          </Link>
        </div>
      </section>

    </div>
  );
}
