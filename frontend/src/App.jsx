import { useEffect, useRef, useState } from "react";
import { Activity, ArrowDownToLine, ArrowRight, BrainCircuit, Check, ChevronDown, CircleHelp, Clock3, FileImage, FileText, HeartPulse, ImagePlus, Layers3, Menu, Microscope, MoreHorizontal, ScanLine, ShieldCheck, SlidersHorizontal, Sparkles, UploadCloud, X } from "lucide-react";

const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
const scanTypes = [
  { value: "chest-xray", label: "Chest X-ray", hint: "Thoracic imaging" },
  { value: "brain-mri", label: "Brain MRI", hint: "Neuro imaging" },
  { value: "skin-lesion", label: "Skin lesion", hint: "Dermatology imaging" },
  { value: "other", label: "Other / not sure", hint: "General image" },
];

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [scanType, setScanType] = useState("chest-xray");
  const [patientId, setPatientId] = useState("");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [mobileNav, setMobileNav] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  function chooseFile(next) {
    if (!next) return;
    setError(""); setResult(null);
    if (!next.type.startsWith("image/")) { setError("Please select a supported image file."); return; }
    if (next.size > 15 * 1024 * 1024) { setError("File is larger than 15 MB. Choose a smaller image."); return; }
    if (preview) URL.revokeObjectURL(preview);
    setFile(next); setPreview(URL.createObjectURL(next));
  }

  async function analyze() {
    if (!file) { setError("Upload a medical image before starting analysis."); return; }
    if (!API_URL) { setError("The prediction API is not configured yet. Set VITE_API_URL to your backend URL to enable real model inference."); return; }
    setLoading(true); setError(""); setResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("scan_type", scanType);
      if (patientId.trim()) body.append("patient_id", patientId.trim());
      const response = await fetch(`${API_URL}/api/predict`, { method: "POST", body });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.message || `Request failed (${response.status})`);
      setResult(data);
    } catch (e) {
      setError(e.message || "Could not reach the prediction service. Check the backend URL and try again.");
    } finally { setLoading(false); }
  }

  function reset() {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null); setPreview(""); setResult(null); setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  const confidence = result?.confidence == null ? null : Math.round(Number(result.confidence) <= 1 ? Number(result.confidence) * 100 : Number(result.confidence));
  const prediction = result?.prediction || result?.predicted_class || result?.label || "Prediction returned";

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
      <div className="brand"><div className="brand-mark"><Activity size={22}/></div><div><strong>medvision<span>AI</span></strong><small>RESEARCH WORKSPACE</small></div><button className="icon-button close-nav" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={18}/></button></div>
      <div className="workspace-label">WORKSPACE</div>
      <nav className="side-nav">
        <a className="nav-item active" href="#overview" onClick={() => setMobileNav(false)}><Layers3 size={18}/> Overview</a>
        <a className="nav-item" href="#analyze" onClick={() => setMobileNav(false)}><ScanLine size={18}/> Image analysis <span className="nav-dot"/></a>
        <a className="nav-item" href="#results" onClick={() => setMobileNav(false)}><FileText size={18}/> Reports</a>
      </nav>
      <div className="workspace-label tools-label">RESEARCH TOOLS</div>
      <nav className="side-nav">
        <a className="nav-item" href="#model"><BrainCircuit size={18}/> CNN model</a>
        <a className="nav-item" href="#safety"><ShieldCheck size={18}/> Safety & privacy</a>
      </nav>
      <div className="sidebar-bottom">
        <div className="help-card"><div className="help-icon"><CircleHelp size={18}/></div><strong>Need a hand?</strong><p>Review the setup guide to connect your model API.</p><a href="https://github.com/Praveen5722/computer-aided-medical-image-diagnosis" target="_blank" rel="noreferrer">Open project docs <ArrowRight size={14}/></a></div>
        <div className="profile"><div className="avatar">MV</div><div><strong>Research workspace</strong><small>Local session</small></div><MoreHorizontal size={19} className="profile-more"/></div>
      </div>
    </aside>

    <main className="main-content" id="overview">
      <header className="topbar"><button className="icon-button menu-button" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={20}/></button><div className="breadcrumbs">Workspace <span>/</span> <b>Image analysis</b></div><div className="top-actions"><span className="status-pill"><i/> Research mode</span><button className="icon-button" aria-label="Help"><CircleHelp size={19}/></button><div className="avatar top-avatar">MV</div></div></header>
      <div className="page-wrap">
        <section className="welcome-row"><div><div className="eyebrow"><Sparkles size={14}/> COMPUTER-AIDED DIAGNOSIS</div><h1>Medical image analysis</h1><p className="subheading">Explore image inputs and connect your regularized CNN model for analysis.</p></div><div className="date-chip"><Clock3 size={15}/> Research session <ChevronDown size={14}/></div></section>
        <section className="notice-banner"><div className="notice-icon"><ShieldCheck size={19}/></div><div><strong>Research tool — not a clinical diagnosis</strong><p>Predictions are experimental and must be reviewed by a qualified medical professional. Do not use for treatment decisions.</p></div><button className="banner-close" aria-label="Dismiss notice" onClick={e => e.currentTarget.parentElement.remove()}><X size={17}/></button></section>

        <section className="metric-grid" aria-label="Workspace summary">
          <div className="metric-card"><div className="metric-top"><span>Analysis pipeline</span><span className="metric-icon blue"><BrainCircuit size={17}/></span></div><div className="metric-value">Regularized CNN</div><div className="metric-foot"><span className="tiny-dot green-dot"/> Model integration ready</div></div>
          <div className="metric-card"><div className="metric-top"><span>Current session</span><span className="metric-icon purple"><ScanLine size={17}/></span></div><div className="metric-value">{file ? "1 image selected" : "No image yet"}</div><div className="metric-foot">Files stay in this browser until submitted</div></div>
          <div className="metric-card"><div className="metric-top"><span>Prediction service</span><span className="metric-icon amber"><Activity size={17}/></span></div><div className="metric-value">{API_URL ? "Configured" : "Not connected"}</div><div className="metric-foot"><span className={`tiny-dot ${API_URL ? "green-dot" : "amber-dot"}`}/>{API_URL ? "API URL provided" : "Set VITE_API_URL to enable"}</div></div>
        </section>

        <div className="section-heading" id="analyze"><div><h2>Start an analysis</h2><p>Upload an image, choose the scan category, and send it to your model API.</p></div><span className="step-label"><span>01</span> INPUT</span></div>
        <section className="analysis-grid">
          <div className="panel upload-panel"><div className="panel-heading"><div className="panel-title-icon"><ImagePlus size={18}/></div><div><h3>Image input</h3><p>Choose a scan image to begin</p></div><button className="icon-button push-right" aria-label="More upload options"><MoreHorizontal size={19}/></button></div>
            <input ref={inputRef} className="file-input" type="file" accept="image/png,image/jpeg,image/webp,image/bmp" onChange={e => chooseFile(e.target.files?.[0])}/>
            {preview ? <div className="preview-wrap"><img src={preview} alt="Selected scan preview"/><div className="preview-overlay"><span><Check size={14}/> Image selected</span><button onClick={reset} aria-label="Remove image"><X size={16}/></button></div><div className="file-details"><div className="filetype-icon"><FileImage size={19}/></div><div className="file-name"><strong>{file.name}</strong><small>{(file.size / (1024 * 1024)).toFixed(2)} MB · {file.type || "Image"}</small></div><span className="file-ready"><Check size={13}/> Ready</span></div></div>
            : <button className={`dropzone ${dragging ? "dragging" : ""}`} onClick={() => inputRef.current?.click()} onDragOver={e => {e.preventDefault();setDragging(true)}} onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault();setDragging(false);chooseFile(e.dataTransfer.files?.[0])}}><div className="upload-symbol"><UploadCloud size={25}/></div><strong>Drop your image here, or <span>browse files</span></strong><p>PNG, JPG, WEBP or BMP · Max 15 MB</p><div className="dropzone-footer"><ShieldCheck size={14}/> Image preview stays in your browser until analysis</div></button>}
            <div className="upload-hint"><span className="hint-dot"/> Use de-identified research images only. Remove personal information before upload.</div>
          </div>
          <div className="panel settings-panel"><div className="panel-heading"><div className="panel-title-icon purple-bg"><SlidersHorizontal size={18}/></div><div><h3>Analysis settings</h3><p>Configure the model input</p></div></div>
            <label className="field-label" htmlFor="scan-type">Image category <span>*</span></label><div className="scan-options">{scanTypes.map(item => <button key={item.value} className={`scan-option ${scanType === item.value ? "selected" : ""}`} onClick={() => setScanType(item.value)}><span className="scan-radio">{scanType === item.value && <i/>}</span><span className="scan-copy"><strong>{item.label}</strong><small>{item.hint}</small></span>{item.value === "chest-xray" ? <HeartPulse size={17}/> : item.value === "brain-mri" ? <BrainCircuit size={17}/> : item.value === "skin-lesion" ? <Microscope size={17}/> : <FileImage size={17}/>}</button>)}</div>
            <label className="field-label patient-label" htmlFor="patient-id">Research reference <span className="optional">(optional)</span></label><input id="patient-id" className="text-input" placeholder="e.g. STUDY-001 (no personal data)" value={patientId} onChange={e => setPatientId(e.target.value)} maxLength={64}/>
            <div className="model-info"><div className="model-info-icon"><BrainCircuit size={18}/></div><div><strong>Regularized Convolutional Neural Network</strong><p>Predictions are provided by your configured backend. This interface does not simulate or invent model results.</p></div></div>
            <button className="primary-button" onClick={analyze} disabled={loading || !file}>{loading ? <><span className="spinner"/> Sending to model…</> : <>Run image analysis <ArrowRight size={17}/></>}</button>
            {error && <div className="error-message" role="alert"><CircleHelp size={16}/><span>{error}</span></div>}
          </div>
        </section>

        <section className="panel result-panel" id="results"><div className="panel-heading result-heading"><div className="panel-title-icon green-bg"><Activity size={18}/></div><div><h3>Analysis results</h3><p>Model response and confidence details appear here</p></div><span className="waiting-pill">{result ? "Response received" : "Awaiting analysis"}</span></div>
          {result ? <div className="result-content"><div className="result-main"><span className="result-kicker">MODEL PREDICTION</span><h2>{String(prediction)}</h2><p>{result.message || "Returned by the connected model service. This is an experimental output, not a clinical conclusion."}</p></div>{confidence !== null && <div className="confidence-box"><span>Model confidence</span><strong>{Math.max(0,Math.min(100,confidence))}%</strong><div className="confidence-track"><i style={{width:`${Math.max(0,Math.min(100,confidence))}%`}}/></div></div>}<button className="secondary-button" onClick={() => {const blob=new Blob([JSON.stringify({prediction:result,scanType,reference:patientId||null,fileName:file?.name,generatedAt:new Date().toISOString(),disclaimer:"Research output only; not a clinical diagnosis."},null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="medvision-analysis.json";a.click();URL.revokeObjectURL(url)}}><ArrowDownToLine size={16}/> Export response</button></div>
          : <div className="empty-results"><div className="empty-illustration"><div className="empty-circle"><ScanLine size={26}/></div><span className="orbit orbit-one"/><span className="orbit orbit-two"/></div><strong>Your results will appear here</strong><p>After you submit an image to a configured prediction API, its response will be shown in this panel.</p><div className="result-steps"><span><i>1</i> Upload scan</span><ArrowRight size={14}/><span><i>2</i> Run analysis</span><ArrowRight size={14}/><span><i>3</i> Review response</span></div></div>}
        </section>
        <section className="bottom-grid"><div className="info-card" id="model"><div className="info-icon purple-bg"><BrainCircuit size={19}/></div><div><h3>Model pipeline</h3><p>Image input → preprocessing → regularized CNN → backend prediction response.</p><a href="https://github.com/Praveen5722/computer-aided-medical-image-diagnosis" target="_blank" rel="noreferrer">View source repository <ArrowRight size={14}/></a></div></div><div className="info-card" id="safety"><div className="info-icon green-bg"><ShieldCheck size={19}/></div><div><h3>Privacy by design</h3><p>Use de-identified scans. Configure HTTPS and appropriate access controls before deployment.</p><a href="https://www.hhs.gov/hipaa/for-professionals/privacy/index.html" target="_blank" rel="noreferrer">Read privacy guidance <ArrowRight size={14}/></a></div></div></section>
        <footer className="footer"><span>© {new Date().getFullYear()} MedVision AI · Research interface</span><span><span className="tiny-dot amber-dot"/> Experimental software · Not for clinical use</span></footer>
      </div>
    </main>
  </div>;
}

export default App;
