"use client";
import { useState, useEffect, useMemo } from "react";

function mmrSelect(qSim: number[], ddSim: number[][], k: number, lambda: number): number[] {
  const n = qSim.length;
  if (n === 0 || k <= 0) return [];
  k = Math.min(k, n);
  let first = 0;
  for (let i = 1; i < n; i++) if (qSim[i] > qSim[first]) first = i;
  const selected = [first];
  const maxSim = [...ddSim[first]];
  const cand = new Array(n).fill(true);
  cand[first] = false;
  while (selected.length < k) {
    let best = -1, bestScore = -Infinity;
    for (let i = 0; i < n; i++) {
      if (!cand[i]) continue;
      const score = lambda * qSim[i] - (1 - lambda) * maxSim[i];
      if (score > bestScore) { bestScore = score; best = i; }
    }
    selected.push(best);
    cand[best] = false;
    for (let i = 0; i < n; i++) maxSim[i] = Math.max(maxSim[i], ddSim[best][i]);
  }
  return selected;
}

function topK(qSim: number[], k: number): number[] {
  return qSim.map((v, i) => [v, i] as [number, number])
    .sort((a, b) => b[0] - a[0]).slice(0, k).map(x => x[1]);
}
function redundancy(ddSim: number[][], sel: number[]): number {
  if (sel.length < 2) return 0;
  let sum = 0, c = 0;
  for (let i = 0; i < sel.length; i++) for (let j = i + 1; j < sel.length; j++) { sum += ddSim[sel[i]][sel[j]]; c++; }
  return sum / c;
}
function maxPair(ddSim: number[][], sel: number[]): number {
  if (sel.length < 2) return 0;
  let m = 0;
  for (let i = 0; i < sel.length; i++) for (let j = i + 1; j < sel.length; j++) m = Math.max(m, ddSim[sel[i]][sel[j]]);
  return m;
}
function meanRel(qSim: number[], sel: number[]): number {
  if (!sel.length) return 0;
  return sel.reduce((a, i) => a + qSim[i], 0) / sel.length;
}

interface Doc { id: number; title: string; category: string; }
interface Data { query: string; documents: Doc[]; query_doc_sim: number[]; doc_doc_sim: number[][]; }

const CAT_COLOR: Record<string, string> = {
  "things-to-do": "#F15153", photography: "#013E37", shopping: "#321847",
  culture: "#B5651D", weather: "#2A6F97", outdoors: "#1B7A43",
  landmarks: "#9A3B8C", food: "#C1440E", transport: "#3D348B",
  history: "#5C4033", offtopic: "#888888",
};

export default function Home() {
  const [data, setData] = useState<Data | null>(null);
  const [lambda, setLambda] = useState(0.7);
  const [k, setK] = useState(7);

  useEffect(() => {
    fetch(`${process.env.NODE_ENV === "production" ? "/mmr-retrieval-lab" : ""}/data.json`)
      .then(r => r.json()).then(setData).catch(() => {});
  }, []);

  const result = useMemo(() => {
    if (!data) return null;
    const mmrIdx = mmrSelect(data.query_doc_sim, data.doc_doc_sim, k, lambda);
    const baseIdx = topK(data.query_doc_sim, k);
    return {
      mmrIdx, baseIdx, baseSet: new Set(baseIdx),
      mmrMetrics: { rel: meanRel(data.query_doc_sim, mmrIdx), red: redundancy(data.doc_doc_sim, mmrIdx), max: maxPair(data.doc_doc_sim, mmrIdx) },
      baseMetrics: { rel: meanRel(data.query_doc_sim, baseIdx), red: redundancy(data.doc_doc_sim, baseIdx), max: maxPair(data.doc_doc_sim, baseIdx) },
    };
  }, [data, lambda, k]);

  if (!data || !result) {
    return <div className="min-h-screen flex items-center justify-center mono" style={{ background: "#FFFDF5" }}>
      <div className="nb-border nb-shadow px-6 py-4 display text-xl" style={{ background: "#FBE311" }}>LOADING…</div>
    </div>;
  }

  const dropPct = result.baseMetrics.max > 0 ? Math.round((1 - result.mmrMetrics.max / result.baseMetrics.max) * 100) : 0;

  return (
    <div className="min-h-screen" style={{ background: "#FFFDF5" }}>
      <header className="nb-border" style={{ borderWidth: "0 0 3px 0", background: "#FBE311" }}>
        <div className="max-w-6xl mx-auto px-5 py-4 flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="display text-2xl sm:text-3xl leading-none">MMR RETRIEVAL LAB</h1>
            <p className="mono text-xs mt-1.5" style={{ color: "#013E37" }}>maximal marginal relevance · diversity vs relevance</p>
          </div>
          <a href="https://github.com/aman-24052001/mmr-retrieval-lab" target="_blank" rel="noreferrer"
            className="nb-border nb-shadow-sm nb-press px-4 py-2 mono text-sm font-bold inline-block" style={{ background: "#FFFDF5" }}>
            VIEW SOURCE →
          </a>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-5 py-6">
        <section className="mb-6">
          <div className="nb-card p-5" style={{ background: "#321847" }}>
            <p className="mono text-xs mb-2" style={{ color: "#FBE311" }}>THE PROBLEM</p>
            <p className="display text-xl sm:text-2xl leading-tight" style={{ color: "#FFFDF5" }}>
              Rank by relevance alone, and your top results repeat themselves.
            </p>
            <p className="mt-3 text-sm sm:text-base" style={{ color: "#FFFDF5", opacity: 0.85 }}>
              Query <span className="mono nb-border px-2 py-0.5" style={{ background: "#FBE311", color: "#1A1A1A" }}>&quot;{data.query}&quot;</span> across {data.documents.length} documents.
              Drag <span style={{ color: "#FBE311", fontWeight: 700 }}>λ</span> below and watch MMR trade relevance for diversity in real time.
            </p>
          </div>
        </section>

        <section className="nb-card p-5 mb-6" style={{ background: "#FFFDF5" }}>
          <div className="flex items-end justify-between mb-3 flex-wrap gap-3">
            <div>
              <p className="mono text-xs mb-1" style={{ color: "#1A1A1A" }}>LAMBDA — RELEVANCE vs DIVERSITY</p>
              <div className="flex items-baseline gap-3">
                <span className="display text-5xl">{lambda.toFixed(2)}</span>
                <span className="mono text-sm" style={{ color: lambda >= 0.7 ? "#013E37" : lambda <= 0.25 ? "#F15153" : "#321847" }}>
                  {lambda >= 0.85 ? "≈ pure relevance" : lambda <= 0.2 ? "diversity dominates" : lambda >= 0.6 && lambda <= 0.8 ? "production sweet spot" : "balanced"}
                </span>
              </div>
            </div>
            <div className="flex gap-2">
              {[0.0, 0.3, 0.7, 1.0].map(v => (
                <button key={v} onClick={() => setLambda(v)} className="nb-border nb-press mono text-sm font-bold px-3 py-1.5"
                  style={{ background: Math.abs(lambda - v) < 0.001 ? "#FBE311" : "#FFFDF5" }}>{v.toFixed(1)}</button>
              ))}
            </div>
          </div>
          <input type="range" min={0} max={1} step={0.05} value={lambda} onChange={e => setLambda(parseFloat(e.target.value))} />
          <div className="flex justify-between mono text-xs mt-2" style={{ color: "#1A1A1A" }}>
            <span>← 0.0 most diverse</span><span>most relevant 1.0 →</span>
          </div>
          <div className="flex items-center gap-3 mt-4 pt-4" style={{ borderTop: "3px solid #1A1A1A" }}>
            <span className="mono text-xs font-bold">RESULTS (k):</span>
            {[5, 7, 10].map(v => (
              <button key={v} onClick={() => setK(v)} className="nb-border nb-press mono text-sm font-bold px-3 py-1"
                style={{ background: k === v ? "#F15153" : "#FFFDF5", color: k === v ? "#FFFDF5" : "#1A1A1A" }}>{v}</button>
            ))}
          </div>
        </section>

        <section className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
          {[
            { label: "MAX PAIR SIM", base: result.baseMetrics.max, mmr: result.mmrMetrics.max, hero: true },
            { label: "REDUNDANCY", base: result.baseMetrics.red, mmr: result.mmrMetrics.red },
            { label: "MEAN RELEVANCE", base: result.baseMetrics.rel, mmr: result.mmrMetrics.rel },
            { label: "DIVERSITY", base: 1 - result.baseMetrics.red, mmr: 1 - result.mmrMetrics.red },
          ].map(m => (
            <div key={m.label} className="nb-card p-3" style={{ background: m.hero ? "#FBE311" : "#FFFDF5" }}>
              <p className="mono text-xs mb-2" style={{ color: "#1A1A1A" }}>{m.label}</p>
              <span className="display text-2xl">{m.mmr.toFixed(3)}</span>
              <p className="mono text-xs mt-1" style={{ color: "#555" }}>base {m.base.toFixed(3)}</p>
            </div>
          ))}
        </section>

        {dropPct > 0 && (
          <div className="nb-card p-3 mb-6 flex items-center gap-3" style={{ background: "#013E37" }}>
            <span className="display text-3xl" style={{ color: "#FBE311" }}>−{dropPct}%</span>
            <span className="text-sm" style={{ color: "#FFFDF5" }}>worst-case redundancy removed vs pure-relevance ranking, at this λ.</span>
          </div>
        )}

        <section className="grid md:grid-cols-2 gap-5">
          <div>
            <div className="nb-border px-3 py-2 mb-3 inline-block mono text-sm font-bold" style={{ background: "#F15153", color: "#FFFDF5" }}>BASELINE · pure relevance</div>
            <div className="flex flex-col gap-2">
              {result.baseIdx.map((idx, rank) => {
                const doc = data.documents[idx];
                const dup = result.baseIdx.some((other, r2) => r2 < rank && data.doc_doc_sim[idx][other] > 0.5);
                return (
                  <div key={idx} className="nb-border nb-shadow-sm p-3 flex items-center gap-3" style={{ background: "#FFFDF5" }}>
                    <span className="display text-lg w-6 shrink-0">{rank + 1}</span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold leading-tight truncate">{doc.title}</p>
                      <p className="mono text-xs mt-0.5" style={{ color: CAT_COLOR[doc.category] }}>{doc.category} · sim {data.query_doc_sim[idx].toFixed(3)}</p>
                    </div>
                    {dup && <span className="mono text-xs font-bold nb-border px-2 py-0.5 shrink-0" style={{ background: "#F15153", color: "#FFFDF5" }}>DUP</span>}
                  </div>
                );
              })}
            </div>
          </div>
          <div>
            <div className="nb-border px-3 py-2 mb-3 inline-block mono text-sm font-bold" style={{ background: "#013E37", color: "#FFFDF5" }}>MMR · λ={lambda.toFixed(2)}</div>
            <div className="flex flex-col gap-2">
              {result.mmrIdx.map((idx, rank) => {
                const doc = data.documents[idx];
                const isNew = !result.baseSet.has(idx);
                return (
                  <div key={idx} className="nb-border nb-shadow-sm p-3 flex items-center gap-3" style={{ background: isNew ? "#FBE311" : "#FFFDF5" }}>
                    <span className="display text-lg w-6 shrink-0">{rank + 1}</span>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold leading-tight truncate">{doc.title}</p>
                      <p className="mono text-xs mt-0.5" style={{ color: CAT_COLOR[doc.category] }}>{doc.category} · sim {data.query_doc_sim[idx].toFixed(3)}</p>
                    </div>
                    {isNew && <span className="mono text-xs font-bold nb-border px-2 py-0.5 shrink-0" style={{ background: "#013E37", color: "#FFFDF5" }}>NEW</span>}
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="nb-card p-5 mt-6" style={{ background: "#1A1A1A" }}>
          <p className="mono text-xs mb-3" style={{ color: "#FBE311" }}>THE FORMULA</p>
          <p className="mono text-sm sm:text-base leading-relaxed" style={{ color: "#FFFDF5" }}>
            MMR = <span style={{ color: "#FBE311" }}>λ</span> · relevance(doc, query) − <span style={{ color: "#F15153" }}>(1−λ)</span> · max_sim(doc, selected)
          </p>
          <p className="text-sm mt-3" style={{ color: "#FFFDF5", opacity: 0.7 }}>
            At each step, pick the document that&apos;s relevant to the query <em>and</em> least similar to what&apos;s already chosen. Computed live in your browser — no backend.
          </p>
        </section>

        <footer className="mono text-xs text-center mt-8 pb-4" style={{ color: "#555" }}>
          Carbonell &amp; Goldstein 1998 · TF-IDF embeddings · full Python implementation + benchmarks in the repo
        </footer>
      </main>
    </div>
  );
}
