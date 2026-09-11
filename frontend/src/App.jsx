import React, { useState, useRef, useEffect } from "react";
import {
  Upload, Eye, Users, AlertTriangle, CheckCircle2,
  Clock, ChevronRight, ScanEye, ArrowLeft, RefreshCw, XCircle
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";

/* ---------------------------------------------------------
   Config — point this at your running backend
--------------------------------------------------------- */
const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

/* ---------------------------------------------------------
   Design tokens
--------------------------------------------------------- */
const COLORS = {
  ink: "#12242E",
  inkSoft: "#3C4E57",
  paper: "#EDEFEC",
  surface: "#FFFFFF",
  line: "#D8DCD6",
  teal: "#2F6F5E",
  tealSoft: "#E4EFEA",
  amber: "#B4791F",
  amberSoft: "#F6ECDA",
  red: "#A63A2C",
  redSoft: "#F5E4E0",
};

const STAGES = [
  { id: 0, label: "No DR", short: "No DR", color: COLORS.teal, soft: COLORS.tealSoft },
  { id: 1, label: "Mild NPDR", short: "Mild", color: COLORS.teal, soft: COLORS.tealSoft },
  { id: 2, label: "Moderate NPDR", short: "Moderate", color: COLORS.amber, soft: COLORS.amberSoft },
  { id: 3, label: "Severe NPDR", short: "Severe", color: COLORS.amber, soft: COLORS.amberSoft },
  { id: 4, label: "Proliferative DR", short: "PDR", color: COLORS.red, soft: COLORS.redSoft },
];

const FONT_URL = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap";

function stageOf(stageId) {
  return STAGES[stageId] ?? STAGES[0];
}

/* ---------------------------------------------------------
   Small building blocks
--------------------------------------------------------- */
function Badge({ stageId }) {
  const stage = stageOf(stageId);
  return (
    <span
      className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium"
      style={{ background: stage.soft, color: stage.color }}
    >
      <span className="w-2 h-2 rounded-full" style={{ background: stage.color }} />
      {stage.label}
    </span>
  );
}

function ConfidenceDial({ value, color }) {
  const r = 54;
  const c = 2 * Math.PI * r;
  const offset = c - (value / 100) * c;
  return (
    <div className="relative w-36 h-36 shrink-0">
      <svg viewBox="0 0 120 120" className="w-36 h-36 -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke={COLORS.line} strokeWidth="8" />
        <circle
          cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="8"
          strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.8s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="font-mono text-2xl font-semibold" style={{ color: COLORS.ink }}>{value}%</span>
        <span className="text-[11px] uppercase tracking-wide" style={{ color: COLORS.inkSoft }}>confidence</span>
      </div>
    </div>
  );
}

function NavItem({ icon: Icon, label, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className="w-full flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
      style={{
        background: active ? "rgba(255,255,255,0.12)" : "transparent",
        color: active ? "#FFFFFF" : "rgba(255,255,255,0.65)",
      }}
    >
      <Icon size={18} />
      {label}
    </button>
  );
}

function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="flex items-center gap-2 px-4 py-3 rounded-lg text-sm mb-4" style={{ background: COLORS.redSoft, color: COLORS.red }}>
      <XCircle size={16} />
      {message}
    </div>
  );
}

/* ---------------------------------------------------------
   Upload screen — sends a real file to POST /predict
--------------------------------------------------------- */
function UploadScreen({ onAnalyze }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [patientId, setPatientId] = useState("");
  const [patientName, setPatientName] = useState("");
  const [dragging, setDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  function handleFile(f) {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  async function handleAnalyze() {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("patient_id", patientId || `PT-${Date.now().toString().slice(-6)}`);
      formData.append("patient_name", patientName || "Unregistered patient");

      const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`Server error (${res.status}): ${detail.slice(0, 200)}`);
      }
      const data = await res.json();
      onAnalyze({ ...data, previewUrl: preview });
    } catch (err) {
      setError(
        err.message.includes("Failed to fetch")
          ? "Can't reach the backend. Is it running at " + API_BASE + "?"
          : err.message
      );
    } finally {
      setAnalyzing(false);
    }
  }

  return (
    <div className="max-w-3xl">
      <h1 className="text-2xl font-semibold" style={{ color: COLORS.ink }}>New screening</h1>
      <p className="text-sm mt-1" style={{ color: COLORS.inkSoft }}>Upload a fundus photograph to grade diabetic retinopathy severity.</p>

      <ErrorBanner message={error} />

      <div className="grid grid-cols-2 gap-4 mt-6">
        <div>
          <label className="text-xs font-medium uppercase tracking-wide" style={{ color: COLORS.inkSoft }}>Patient ID</label>
          <input
            value={patientId} onChange={(e) => setPatientId(e.target.value)} placeholder="e.g. PT-1042"
            className="mt-1 w-full px-3 py-2 rounded-lg border text-sm outline-none"
            style={{ borderColor: COLORS.line, background: COLORS.surface }}
          />
        </div>
        <div>
          <label className="text-xs font-medium uppercase tracking-wide" style={{ color: COLORS.inkSoft }}>Patient name</label>
          <input
            value={patientName} onChange={(e) => setPatientName(e.target.value)} placeholder="e.g. R. Meena"
            className="mt-1 w-full px-3 py-2 rounded-lg border text-sm outline-none"
            style={{ borderColor: COLORS.line, background: COLORS.surface }}
          />
        </div>
      </div>

      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files[0]); }}
        onClick={() => inputRef.current.click()}
        className="mt-6 rounded-2xl border-2 border-dashed flex flex-col items-center justify-center py-14 cursor-pointer transition-colors"
        style={{ borderColor: dragging ? COLORS.teal : COLORS.line, background: dragging ? COLORS.tealSoft : COLORS.surface }}
      >
        <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => handleFile(e.target.files[0])} />
        {preview ? (
          <div className="flex flex-col items-center gap-3">
            <img src={preview} alt="Fundus preview" className="w-40 h-40 object-cover rounded-full border-4" style={{ borderColor: COLORS.paper }} />
            <span className="text-sm" style={{ color: COLORS.inkSoft }}>Click or drop to replace image</span>
          </div>
        ) : (
          <>
            <div className="w-14 h-14 rounded-full flex items-center justify-center mb-3" style={{ background: COLORS.tealSoft }}>
              <Upload size={22} color={COLORS.teal} />
            </div>
            <p className="text-sm font-medium" style={{ color: COLORS.ink }}>Drop fundus image here, or click to browse</p>
            <p className="text-xs mt-1" style={{ color: COLORS.inkSoft }}>JPG or PNG, single eye per image</p>
          </>
        )}
      </div>

      <button
        disabled={!file || analyzing}
        onClick={handleAnalyze}
        className="mt-6 w-full py-3 rounded-lg text-sm font-semibold flex items-center justify-center gap-2 transition-opacity disabled:opacity-40"
        style={{ background: COLORS.ink, color: "#fff" }}
      >
        {analyzing ? (
          <>
            <RefreshCw size={16} className="animate-spin" /> Analyzing image…
          </>
        ) : (
          <>
            <ScanEye size={16} /> Analyze scan
          </>
        )}
      </button>
    </div>
  );
}

/* ---------------------------------------------------------
   Result screen — shows the real Grad-CAM overlay from the backend
--------------------------------------------------------- */
function ResultScreen({ result, onBack }) {
  const [showOverlay, setShowOverlay] = useState(true);
  const stage = stageOf(result.stage_id);
  const overlaySrc = `${API_BASE}${result.overlay_url}`;

  return (
    <div className="max-w-4xl">
      <button onClick={onBack} className="flex items-center gap-1.5 text-sm font-medium mb-4" style={{ color: COLORS.inkSoft }}>
        <ArrowLeft size={15} /> Back to upload
      </button>

      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-semibold" style={{ color: COLORS.ink }}>Screening result</h1>
          <p className="text-sm mt-1 font-mono" style={{ color: COLORS.inkSoft }}>{result.patient_id} · scan #{result.scan_id}</p>
        </div>
        <Badge stageId={result.stage_id} />
      </div>

      <div className="grid grid-cols-5 gap-6 mt-6">
        <div className="col-span-3 rounded-2xl p-5" style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}` }}>
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium uppercase tracking-wide" style={{ color: COLORS.inkSoft }}>Retinal image</span>
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className="text-xs font-medium px-3 py-1.5 rounded-full"
              style={{ background: showOverlay ? COLORS.tealSoft : COLORS.paper, color: showOverlay ? COLORS.teal : COLORS.inkSoft }}
            >
              {showOverlay ? "Hide" : "Show"} Grad-CAM overlay
            </button>
          </div>
          <div className="relative w-full aspect-square rounded-xl overflow-hidden" style={{ background: "#000" }}>
            <img src={showOverlay ? overlaySrc : result.previewUrl} alt="Fundus scan" className="w-full h-full object-cover" />
          </div>
          <p className="text-xs mt-3" style={{ color: COLORS.inkSoft }}>
            Highlighted regions indicate areas that most influenced the model's grading.
          </p>
        </div>

        <div className="col-span-2 flex flex-col gap-4">
          <div className="rounded-2xl p-5 flex flex-col items-center" style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}` }}>
            <ConfidenceDial value={result.confidence} color={stage.color} />
            {result.low_confidence_flag && (
              <div className="flex items-center gap-1.5 mt-3 text-xs font-medium" style={{ color: COLORS.amber }}>
                <AlertTriangle size={13} /> Low confidence — recommend manual review
              </div>
            )}
          </div>

          <div className="rounded-2xl p-5" style={{ background: stage.soft, border: `1px solid ${stage.color}33` }}>
            <span className="text-xs font-medium uppercase tracking-wide" style={{ color: stage.color }}>Recommended action</span>
            <p className="text-sm font-medium mt-1.5" style={{ color: COLORS.ink }}>{result.recommended_action}</p>
          </div>

          <div className="rounded-2xl p-5" style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}` }}>
            <span className="text-xs font-medium uppercase tracking-wide" style={{ color: COLORS.inkSoft }}>Class probabilities</span>
            <div className="flex flex-col gap-2 mt-3">
              {Object.entries(result.class_probabilities || {}).map(([name, p]) => (
                <div key={name} className="flex items-center justify-between text-xs">
                  <span style={{ color: COLORS.inkSoft }}>{name}</span>
                  <span className="font-mono font-medium" style={{ color: COLORS.ink }}>{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------
   Patients screens — pulled live from GET /patients
--------------------------------------------------------- */
function PatientsScreen({ onSelect, refreshKey }) {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/patients`)
      .then((r) => { if (!r.ok) throw new Error(`Server error (${r.status})`); return r.json(); })
      .then((data) => { setPatients(data); setError(null); })
      .catch((err) => setError("Couldn't load patients — is the backend running?"))
      .finally(() => setLoading(false));
  }, [refreshKey]);

  return (
    <div className="max-w-4xl">
      <h1 className="text-2xl font-semibold" style={{ color: COLORS.ink }}>Patients</h1>
      <p className="text-sm mt-1" style={{ color: COLORS.inkSoft }}>
        {loading ? "Loading…" : `${patients.length} patients screened at this clinic`}
      </p>

      <ErrorBanner message={error} />

      {!loading && patients.length === 0 && !error && (
        <div className="mt-6 text-sm rounded-2xl p-8 text-center" style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}`, color: COLORS.inkSoft }}>
          No patients yet — run a screening to see them appear here.
        </div>
      )}

      {patients.length > 0 && (
        <div className="mt-6 rounded-2xl overflow-hidden" style={{ border: `1px solid ${COLORS.line}`, background: COLORS.surface }}>
          <table className="w-full text-sm">
            <thead>
              <tr style={{ background: COLORS.paper }}>
                {["Patient", "Age / Sex", "Last scan", "Latest grade", ""].map((h) => (
                  <th key={h} className="text-left font-medium px-5 py-3 text-xs uppercase tracking-wide" style={{ color: COLORS.inkSoft }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {patients.map((p) => {
                const latest = p.scans?.[p.scans.length - 1];
                return (
                  <tr
                    key={p.id} onClick={() => onSelect(p)}
                    className="cursor-pointer hover:opacity-80"
                    style={{ borderTop: `1px solid ${COLORS.line}` }}
                  >
                    <td className="px-5 py-3.5">
                      <div className="font-medium" style={{ color: COLORS.ink }}>{p.name}</div>
                      <div className="text-xs font-mono" style={{ color: COLORS.inkSoft }}>{p.id}</div>
                    </td>
                    <td className="px-5 py-3.5" style={{ color: COLORS.inkSoft }}>{p.age ?? "—"} · {p.sex ?? "—"}</td>
                    <td className="px-5 py-3.5" style={{ color: COLORS.inkSoft }}>
                      {latest ? new Date(latest.created_at).toLocaleDateString() : "—"}
                    </td>
                    <td className="px-5 py-3.5">{latest ? <Badge stageId={latest.stage_id} /> : "—"}</td>
                    <td className="px-5 py-3.5 text-right pr-5"><ChevronRight size={16} color={COLORS.inkSoft} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function PatientDetailScreen({ patient, onBack }) {
  const chartData = (patient.scans || []).map((s) => ({
    d: new Date(s.created_at).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
    s: s.stage_id,
  }));
  const latest = patient.scans?.[patient.scans.length - 1];

  return (
    <div className="max-w-3xl">
      <button onClick={onBack} className="flex items-center gap-1.5 text-sm font-medium mb-4" style={{ color: COLORS.inkSoft }}>
        <ArrowLeft size={15} /> Back to patients
      </button>
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold" style={{ color: COLORS.ink }}>{patient.name}</h1>
          <p className="text-sm mt-1 font-mono" style={{ color: COLORS.inkSoft }}>{patient.id} · {patient.age ?? "—"}y · {patient.sex ?? "—"}</p>
        </div>
        {latest && <Badge stageId={latest.stage_id} />}
      </div>

      {chartData.length > 1 && (
        <div className="mt-6 rounded-2xl p-5" style={{ background: COLORS.surface, border: `1px solid ${COLORS.line}` }}>
          <span className="text-xs font-medium uppercase tracking-wide" style={{ color: COLORS.inkSoft }}>Severity progression</span>
          <div className="h-52 mt-3">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid stroke={COLORS.line} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="d" tick={{ fontSize: 12, fill: COLORS.inkSoft }} axisLine={{ stroke: COLORS.line }} tickLine={false} />
                <YAxis domain={[0, 4]} ticks={[0, 1, 2, 3, 4]} tickFormatter={(v) => STAGES[v].short} tick={{ fontSize: 11, fill: COLORS.inkSoft }} axisLine={false} tickLine={false} width={80} />
                <Tooltip
                  formatter={(v) => STAGES[v].label}
                  contentStyle={{ borderRadius: 10, border: `1px solid ${COLORS.line}`, fontSize: 12, fontFamily: "IBM Plex Mono, monospace" }}
                />
                <Line type="monotone" dataKey="s" stroke={COLORS.teal} strokeWidth={2.5} dot={{ r: 5, fill: COLORS.teal }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="mt-4 rounded-2xl divide-y" style={{ border: `1px solid ${COLORS.line}`, background: COLORS.surface }}>
        {(patient.scans || []).slice().reverse().map((s) => (
          <div key={s.id} className="flex items-center justify-between px-5 py-3.5">
            <div className="flex items-center gap-3">
              <Clock size={15} color={COLORS.inkSoft} />
              <span className="text-sm" style={{ color: COLORS.ink }}>{new Date(s.created_at).toLocaleString()}</span>
              <span className="text-xs font-mono" style={{ color: COLORS.inkSoft }}>{s.confidence}% confidence</span>
            </div>
            <Badge stageId={s.stage_id} />
          </div>
        ))}
        {(!patient.scans || patient.scans.length === 0) && (
          <div className="px-5 py-6 text-sm text-center" style={{ color: COLORS.inkSoft }}>No scans recorded yet.</div>
        )}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------
   App shell
--------------------------------------------------------- */
export default function App() {
  const [view, setView] = useState("upload");
  const [result, setResult] = useState(null);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientsRefreshKey, setPatientsRefreshKey] = useState(0);

  function handleAnalyze(apiResult) {
    setResult(apiResult);
    setView("result");
    setPatientsRefreshKey((k) => k + 1); // so the patients list picks up the new scan
  }

  const nav = [
    { key: "upload", label: "New Scan", icon: ScanEye },
    { key: "patients", label: "Patients", icon: Users },
  ];

  return (
    <div className="w-full min-h-screen flex" style={{ background: COLORS.paper, fontFamily: "'IBM Plex Sans', sans-serif" }}>
      <style>{`
        @import url('${FONT_URL}');
        .font-mono { font-family: 'IBM Plex Mono', monospace; }
      `}</style>

      <aside className="w-60 shrink-0 flex flex-col py-6 px-4" style={{ background: COLORS.ink }}>
        <div className="flex items-center gap-2 px-2 mb-8">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: COLORS.teal }}>
            <Eye size={16} color="#fff" />
          </div>
          <div>
            <div className="text-sm font-semibold text-white leading-tight">RetinaCheck</div>
            <div className="text-[11px] leading-tight" style={{ color: "rgba(255,255,255,0.5)" }}>Clinic screening</div>
          </div>
        </div>

        <nav className="flex flex-col gap-1">
          {nav.map((n) => (
            <NavItem
              key={n.key} icon={n.icon} label={n.label}
              active={view === n.key || (n.key === "patients" && view === "patientDetail")}
              onClick={() => { setView(n.key); setSelectedPatient(null); }}
            />
          ))}
        </nav>

        <div className="mt-auto px-4 py-3 rounded-lg" style={{ background: "rgba(255,255,255,0.06)" }}>
          <div className="text-xs font-medium text-white">Aravind Rural Clinic</div>
          <div className="text-[11px] mt-0.5" style={{ color: "rgba(255,255,255,0.5)" }}>Technician · S. Kumar</div>
        </div>
      </aside>

      <main className="flex-1 px-10 py-8 overflow-y-auto">
        {view === "upload" && <UploadScreen onAnalyze={handleAnalyze} />}
        {view === "result" && result && (
          <ResultScreen result={result} onBack={() => setView("upload")} />
        )}
        {view === "patients" && (
          <PatientsScreen refreshKey={patientsRefreshKey} onSelect={(p) => { setSelectedPatient(p); setView("patientDetail"); }} />
        )}
        {view === "patientDetail" && selectedPatient && (
          <PatientDetailScreen patient={selectedPatient} onBack={() => setView("patients")} />
        )}
      </main>
    </div>
  );
}
