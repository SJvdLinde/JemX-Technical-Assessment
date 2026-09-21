"use client";

/*
 * The ops room.
 *
 * Four tabs, one per requirement: who goes over, why the hours happen, which
 * shifts have no clock-out, and loading next week. Figures over prose -- the
 * reader is on a phone between sites with about ten minutes.
 */

import { useCallback, useEffect, useState } from "react";
import { fetchAnalysis, hrs, rand, uploadExport } from "@/lib/api";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const CAP = 55;

export default function Page() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(true);
  const [tab, setTab] = useState("risk");

  useEffect(() => {
    fetchAnalysis().then(setData).catch((e) => setError(e.message)).finally(() => setBusy(false));
  }, []);

  const onUpload = useCallback(async (event) => {
    const files = event.target.files;
    if (!files?.length) return;
    setBusy(true);
    setError(null);
    try {
      setData(await uploadExport(files));
      setTab("risk");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      event.target.value = "";
    }
  }, []);

  if (busy && !data) return <main><p className="sub">Loading…</p></main>;
  if (error && !data) return <main><div className="panel err">{error}</div></main>;
  if (!data) return null;

  const { week, period, headline, actions, systemic, reasons, sites, data_gaps } = data;
  const cutoff = DAYS[week.days_elapsed - 1];
  const needing = actions.filter((a) => a.needs_action).length;

  const tabs = [
    { id: "risk", label: "At risk", badge: needing },
    { id: "why", label: "Why", badge: null },
    { id: "gaps", label: "No clock-out", badge: data_gaps.unclosed_shifts.length || null },
    { id: "load", label: "Load week", badge: null },
  ];

  return (
    <main>
      <h1>Who goes over by Sunday</h1>
      <p className="sub">
        {week.start} → {week.end} · data to {cutoff} {week.data_through} ·{" "}
        {week.days_remaining} days left
      </p>

      {error && <div className="panel err">{error}</div>}
      {data.warnings?.length > 0 && (
        <div className="panel warnbox">
          {data.warnings.map((w, i) => <div key={i}>{w}</div>)}
        </div>
      )}

      <div className="stats">
        <Stat n={needing} label="need a change" />
        <Stat n={headline.employees} label="employees" />
        <Stat n={hrs(headline.hours_so_far)} label={`worked Mon–${cutoff}`} />
        <Stat n={rand(headline.premium_cost_so_far)} label="premium pay" />
      </div>

      <nav className="tabs">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? "tab on" : "tab"}
            onClick={() => setTab(t.id)}
          >
            {t.label}
            {t.badge ? <span className="badge">{t.badge}</span> : null}
          </button>
        ))}
      </nav>

      {tab === "risk" && actions.map((a) => <Person key={a.employee_id} a={a} />)}

      {tab === "why" && (
        <>
          <div className="kpis">
            <KPI n={headline.notes_operational_failure} l="failures we pay for" tone="bad" />
            <KPI n={headline.notes_client_billable} l="billable to client" tone="ok" />
            <KPI n={headline.notes_total} l="notes read" />
            <KPI n={`${period.weeks}w`} l={`to ${period.to}`} />
          </div>

          {systemic.map((s) => (
            <div className="panel" key={s.category}>
              <div className="row">
                <span className="name">{s.category.replace(/_/g, " ")}</span>
                <span className="figure">{rand(s.cost)}</span>
              </div>
              <div className="meta">
                {Math.round(s.share_of_avoidable * 100)}% of avoidable · {s.notes} notes ·{" "}
                {hrs(s.hours)}
              </div>
              <div className="meta">
                Worst: {s.worst_site_name} {rand(s.worst_site_cost)}
              </div>
              <div className="fix">{s.fix}</div>
            </div>
          ))}

          <div className="panel">
            <table>
              <thead>
                <tr>
                  <th>Reason</th><th className="num">Notes</th>
                  <th className="num">Cost</th><th>Pays</th>
                </tr>
              </thead>
              <tbody>
                {reasons.map((r) => (
                  <tr key={r.category}>
                    <td>{r.category.replace(/_/g, " ")}</td>
                    <td className="num">{r.notes}</td>
                    <td className="num">{rand(r.cost)}</td>
                    <td>
                      <span className={`tag ${r.who_pays === "company_pays" ? "company" : r.who_pays === "client_pays" ? "client" : ""}`}>
                        {r.who_pays === "company_pays" ? "us" : r.who_pays === "client_pays" ? "client" : "—"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="panel">
            <table>
              <thead>
                <tr>
                  <th>Site</th><th className="num">Avoidable</th><th>Biggest cause</th>
                </tr>
              </thead>
              <tbody>
                {sites.map((s) => (
                  <tr key={s.site_id}>
                    <td>{s.site_name}</td>
                    <td className="num">{rand(s.avoidable_cost)}</td>
                    <td className="meta">{s.top_cause?.replace(/_/g, " ") ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {tab === "gaps" && (
        <>
          <div className="kpis">
            <KPI n={data_gaps.unclosed_shifts.length} l="no clock-out this week" tone="bad" />
            <KPI n={data_gaps.unclear_notes} l="notes uncategorised" />
            <KPI n={data_gaps.capped_shifts} l="shifts at the 13.5h ceiling" />
          </div>
          <div className="panel">
            {data_gaps.unclosed_shifts.length === 0 ? (
              <div className="meta">Every shift this week has a clock-out.</div>
            ) : (
              <table>
                <thead>
                  <tr><th>Employee</th><th className="num">Est.</th><th>Note</th></tr>
                </thead>
                <tbody>
                  {data_gaps.unclosed_shifts.map((u) => (
                    <tr key={u.shift_id}>
                      <td>{u.full_name}</td>
                      <td className="num">{hrs(u.hours)}</td>
                      <td className="note">{u.note || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
          {data_gaps.suggested_categories?.length > 0 && (
            <div className="panel">
              <div className="meta">New causes suggested</div>
              {data_gaps.suggested_categories.map((s) => (
                <div className="row" key={s.name}>
                  <span>{s.name.replace(/_/g, " ")}</span>
                  <span className="figure">{s.count}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {tab === "load" && (
        <div className="panel">
          <div className="meta" style={{ marginBottom: 12 }}>
            Seven CSVs. Week, cut-off day and model are read from the files.
          </div>
          <label className="upload">
            {busy ? "Working…" : "Choose CSV files"}
            <input type="file" multiple accept=".csv" onChange={onUpload} disabled={busy} />
          </label>
        </div>
      )}
    </main>
  );
}

function Stat({ n, label }) {
  return <div className="stat"><div className="n">{n}</div><div className="l">{label}</div></div>;
}

function KPI({ n, l, tone }) {
  return (
    <div className={`kpi ${tone || ""}`}>
      <div className="n">{n}</div>
      <div className="l">{l}</div>
    </div>
  );
}

function Person({ a }) {
  const band = a.risk_score >= 0.5 ? "high" : a.risk_score >= 0.2 ? "mid" : "low";
  const pct = Math.min(100, (a.projected_hours / CAP) * 100);
  const over = a.projected_hours > CAP;

  return (
    <div className={`panel person-card ${a.needs_action ? "flagged" : ""}`}>
      <div className="person">
        <div className={`risk ${band}`}>
          {Math.round(a.risk_score * 100)}
          <span className="l">risk</span>
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="row">
            <span className="name">{a.full_name}</span>
            <span className={`figure ${over ? "bad" : ""}`}>{hrs(a.projected_hours)}</span>
          </div>
          <div className="meta">{a.role} · {a.site_name}</div>

          <div className={`bar ${over ? "over" : ""}`}>
            <span style={{ width: `${pct}%` }} />
          </div>

          <div className="figures">
            <Figure v={hrs(a.hours_so_far)} l="worked" />
            <Figure v={hrs(a.projected_hours)} l="projected" />
            <Figure
              v={over ? `+${(a.projected_hours - CAP).toFixed(1)}h` : `${a.headroom_hours.toFixed(1)}h`}
              l={over ? "over cap" : "room left"}
              tone={over ? "bad" : null}
            />
            <Figure v={hrs(a.typical_shift_hours)} l="typical shift" />
          </div>

          <div className="action">{a.action}</div>

          {a.needs_action && (
            a.cover_options.length > 0 ? (
              <div className="cover">
                Cover:{" "}
                {a.cover_options.map((c, i) => (
                  <span key={c.employee_id}>
                    {i > 0 && " · "}
                    <b>{c.full_name}</b> {hrs(c.projected_hours)}→{hrs(c.projected_after_swap)}
                  </span>
                ))}
              </div>
            ) : (
              <div className="cover">No cover available at {a.site_name}.</div>
            )
          )}
        </div>
      </div>
    </div>
  );
}

function Figure({ v, l, tone }) {
  return (
    <div className="fig">
      <div className={`v ${tone || ""}`}>{v}</div>
      <div className="k">{l}</div>
    </div>
  );
}
