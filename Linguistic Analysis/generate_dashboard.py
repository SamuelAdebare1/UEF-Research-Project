"""
Reads results JSON files and generates a self-contained dashboard.html.
Run from inside the Linguistic Analysis folder:
    python3 generate_dashboard.py
"""
import json

with open("results_anomalies.json") as f:    anomalies_data = json.load(f)
with open("results_word_frequency.json") as f: wf_data = json.load(f)

# ── Anomaly chart data (bottom 15 sentences by score) ──────────────────────
top_anomalies = sorted(anomalies_data["anomalies"], key=lambda x: x["similarity_score"])[:15]
anomaly_labels = [f"[{a['index']}] {a['sentence'][:40]}…" for a in top_anomalies]
anomaly_scores = [a["similarity_score"] for a in top_anomalies]
anomaly_colors = ["#ef4444" if s < 0 else "#f97316" if s < 0.15 else "#facc15"
                  for s in anomaly_scores]

# ── Word frequency data ──────────────────────────────────────────────────────
top_words = list(wf_data["top_100_content_words"].items())[:20]
wf_labels = [w for w, _ in top_words]
wf_counts = [c for _, c in top_words]

# ── Hapax legomena ────────────────────────────────────────────────────────────
hapax_list = wf_data.get("hapax_legomena", [])

# ── Embed into HTML ───────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Linguistic Analysis Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0;
    --text: #0f172a; --muted: #64748b;
    --red: #dc2626; --orange: #ea580c; --yellow: #ca8a04;
    --green: #16a34a; --blue: #2563eb; --teal: #0d9488;
  }}
  [data-theme="dark"] {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #f1f5f9; --muted: #94a3b8;
    --red: #ef4444; --orange: #f97316; --yellow: #eab308;
    --green: #22c55e; --blue: #3b82f6; --teal: #14b8a6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: system-ui, sans-serif; padding: 2rem; transition: background 0.2s, color 0.2s; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; }}
  .subtitle {{ color: var(--muted); margin-bottom: 2rem; font-size: 0.9rem; }}
  .header-row {{ display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 2rem; }}
  .toggle-btn {{
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); border-radius: 999px; padding: 0.4rem 1rem;
    font-size: 0.85rem; font-weight: 500; cursor: pointer;
    transition: background 0.2s, border-color 0.2s;
  }}
  .toggle-btn:hover {{ border-color: var(--blue); }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 0.75rem; padding: 1.25rem; transition: background 0.2s, border-color 0.2s; }}
  .stat-card {{ text-align: center; }}
  .stat-card .value {{ font-size: 2rem; font-weight: 800; }}
  .stat-card .label {{ color: var(--muted); font-size: 0.8rem; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  .section-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }}
  .badge {{ font-size: 0.7rem; padding: 0.2rem 0.5rem; border-radius: 999px; font-weight: 600; }}
  .badge-red {{ background: #fee2e2; color: #991b1b; }}
  .badge-blue {{ background: #dbeafe; color: #1e40af; }}
  [data-theme="dark"] .badge-red {{ background: #7f1d1d; color: #fca5a5; }}
  [data-theme="dark"] .badge-blue {{ background: #1e3a5f; color: #93c5fd; }}
  canvas {{ max-height: 320px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ text-align: left; padding: 0.5rem 0.75rem; color: var(--muted); border-bottom: 1px solid var(--border); font-weight: 500; }}
  td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  .score-negative {{ color: var(--red); font-weight: 700; }}
  .score-low {{ color: var(--orange); font-weight: 700; }}
  .score-mid {{ color: var(--yellow); font-weight: 600; }}
  .sent {{ color: var(--muted); font-style: italic; }}
  @media (max-width: 900px) {{ .grid-4 {{ grid-template-columns: 1fr 1fr; }} .grid-2 {{ grid-template-columns: 1fr; }} }}
  .info-wrap {{ position: relative; display: inline-flex; align-items: center; margin-left: 0.35rem; }}
  .info-icon {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.1rem; height: 1.1rem; border-radius: 50%;
    background: var(--blue); color: #fff;
    font-size: 0.65rem; font-weight: 700; cursor: default;
    font-style: normal; line-height: 1; flex-shrink: 0;
  }}
  .tooltip {{
    display: none; position: absolute; bottom: calc(100% + 6px); left: 50%;
    transform: translateX(-50%);
    background: var(--text); color: var(--bg);
    font-size: 0.78rem; font-weight: 400; line-height: 1.45;
    padding: 0.5rem 0.75rem; border-radius: 0.5rem;
    width: 220px; text-align: left; white-space: normal;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15); z-index: 10;
    pointer-events: none;
  }}
  .tooltip::after {{
    content: ''; position: absolute; top: 100%; left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: var(--text);
  }}
  .info-wrap:hover .tooltip {{ display: block; }}
  .hapax-search {{
    width: 100%; padding: 0.5rem 0.75rem; margin-bottom: 1rem;
    border: 1px solid var(--border); border-radius: 0.5rem;
    background: var(--bg); color: var(--text); font-size: 0.85rem;
    outline: none; transition: border-color 0.2s;
  }}
  .hapax-search:focus {{ border-color: var(--blue); }}
  .hapax-cloud {{
    display: flex; flex-wrap: wrap; gap: 0.4rem;
    max-height: 320px; overflow-y: auto;
    padding-right: 0.25rem;
  }}
  .hapax-pill {{
    display: inline-block; padding: 0.2rem 0.6rem;
    background: #eff6ff; color: #1e40af;
    border: 1px solid #bfdbfe; border-radius: 999px;
    font-size: 0.78rem; font-weight: 500;
  }}
  [data-theme="dark"] .hapax-pill {{
    background: #1e3a5f; color: #93c5fd; border-color: #1e40af;
  }}
  .hapax-pill.hidden {{ display: none; }}
  .hapax-none {{ color: var(--muted); font-size: 0.85rem; display: none; }}
</style>
</head>
<body>

<div class="header-row">
  <div>
    <h1>Linguistic Analysis Dashboard</h1>
    <p class="subtitle">Genesis (50-pages.pdf) — injection detection &amp; vocabulary profiling</p>
  </div>
  <button class="toggle-btn" onclick="toggleTheme()" id="themeBtn">🌙 Dark mode</button>
</div>

<!-- ── STAT CARDS ── -->
<div class="grid-4">
  <div class="card stat-card">
    <div class="value" style="color:var(--blue)">{wf_data['total_tokens']:,}</div>
    <div class="label" style="display:inline-flex;align-items:center;justify-content:center">
      Total Tokens
      <span class="info-wrap"><i class="info-icon">i</i><span class="tooltip">The total number of word pieces (tokens) in the document after splitting the text. A token is roughly one word or punctuation mark.</span></span>
    </div>
  </div>
  <div class="card stat-card">
    <div class="value" style="color:var(--teal)">{wf_data['unique_types']:,}</div>
    <div class="label" style="display:inline-flex;align-items:center;justify-content:center">
      Unique Types
      <span class="info-wrap"><i class="info-icon">i</i><span class="tooltip">The number of distinct word forms found in the document. A high count means a rich, varied vocabulary.</span></span>
    </div>
  </div>
  <div class="card stat-card">
    <div class="value" style="color:var(--red)">{len(anomalies_data['anomalies'])}</div>
    <div class="label" style="display:inline-flex;align-items:center;justify-content:center">
      Anomalies Found
      <span class="info-wrap"><i class="info-icon">i</i><span class="tooltip">Sentences whose semantic similarity to their neighbours falls below the threshold (0.3). Low scores signal sentences that are stylistically out of place — likely injected content.</span></span>
    </div>
  </div>
  <div class="card stat-card">
    <div class="value" style="color:var(--yellow)">{wf_data['hapax_legomena_count']:,}</div>
    <div class="label" style="display:inline-flex;align-items:center;justify-content:center">
      Hapax Legomena
      <span class="info-wrap"><i class="info-icon">i</i><span class="tooltip">Words that appear exactly once in the entire document. A high hapax count indicates a wide vocabulary with many rare or unique words.</span></span>
    </div>
  </div>
</div>

<!-- ── ROW 1: Anomaly chart + Word freq ── -->
<div class="grid-2">
  <div class="card">
    <div class="section-title">Semantic Anomalies <span class="badge badge-red">Injection Detection</span><span class="info-wrap"><i class="info-icon">i</i><span class="tooltip">Each sentence is embedded as a vector and compared to its 5 nearest neighbours. Sentences with a mean similarity below 0.3 are flagged — they are semantically isolated from surrounding text.</span></span></div>
    <canvas id="anomalyChart"></canvas>
  </div>
  <div class="card">
    <div class="section-title">Top 20 Content Words <span class="badge badge-blue">Vocabulary</span><span class="info-wrap"><i class="info-icon">i</i><span class="tooltip">Content words (nouns, verbs, adjectives) ranked by how often they appear. Stop words like "the", "and", "of" are excluded. High-frequency words reveal the document's key themes and entities.</span></span></div>
    <canvas id="wordFreqChart"></canvas>
  </div>
</div>

<!-- ── ROW 2: Anomaly table ── -->
<div class="card" style="margin-bottom:2rem">
  <div class="section-title">Top Anomalous Sentences <span class="badge badge-red">Ranked by Score</span></div>
  <table>
    <thead><tr><th>#</th><th>Score</th><th>Sentence (truncated)</th></tr></thead>
    <tbody>
      {"".join(
        f'<tr><td>{a["index"]}</td>'
        f'<td class="{"score-negative" if a["similarity_score"] < 0 else "score-low" if a["similarity_score"] < 0.15 else "score-mid"}">{a["similarity_score"]:.4f}</td>'
        f'<td class="sent">{a["sentence"][:100].strip()}…</td></tr>'
        for a in top_anomalies
      )}
    </tbody>
  </table>
</div>

<script>
const CHART_DEFAULTS = {{
  plugins: {{ legend: {{ labels: {{ color: '#0f172a' }} }} }},
  scales: {{
    x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#e2e8f0' }} }},
    y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#e2e8f0' }} }},
  }}
}};

const anomalyChartObj = new Chart(document.getElementById('anomalyChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(anomaly_labels)},
    datasets: [{{ label: 'Similarity Score', data: {json.dumps(anomaly_scores)},
      backgroundColor: {json.dumps(anomaly_colors)}, borderRadius: 4 }}]
  }},
  options: {{ ...CHART_DEFAULTS, indexAxis: 'y',
    plugins: {{ ...CHART_DEFAULTS.plugins, legend: {{ display: false }} }},
    scales: {{ x: CHART_DEFAULTS.scales.x, y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 10 }} }}, grid: {{ color: '#334155' }} }} }}
  }}
}});

const wordFreqChartObj = new Chart(document.getElementById('wordFreqChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(wf_labels)},
    datasets: [{{ label: 'Count', data: {json.dumps(wf_counts)},
      backgroundColor: '#3b82f6', borderRadius: 4 }}]
  }},
  options: {{ ...CHART_DEFAULTS, indexAxis: 'y',
    plugins: {{ ...CHART_DEFAULTS.plugins, legend: {{ display: false }} }}
  }}
}});
</script>

<!-- ── ROW 3: Hapax Legomena ── -->
<div class="card" style="margin-bottom:2rem">
  <div class="section-title">
    Hapax Legomena
    <span class="badge badge-blue">{len(hapax_list):,} words</span>
    <span class="info-wrap"><i class="info-icon">i</i><span class="tooltip">Words that appear exactly once in the document. These reveal rare, specialised, or injected vocabulary. Use the search box to filter.</span></span>
  </div>
  <input class="hapax-search" id="hapaxSearch" type="search" placeholder="Search hapax legomena…" oninput="filterHapax(this.value)"/>
  <div class="hapax-cloud" id="hapaxCloud">
    {"".join(f'<span class="hapax-pill">{w}</span>' for w in hapax_list)}
  </div>
  <p class="hapax-none" id="hapaxNone">No matches found.</p>
</div>

<script>
  function filterHapax(query) {{
    const q = query.toLowerCase();
    let visible = 0;
    document.querySelectorAll('.hapax-pill').forEach(pill => {{
      const match = pill.textContent.includes(q);
      pill.classList.toggle('hidden', !match);
      if (match) visible++;
    }});
    document.getElementById('hapaxNone').style.display = visible === 0 ? 'block' : 'none';
  }}
</script>

<script>
  function toggleTheme() {{
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    document.documentElement.setAttribute('data-theme', isDark ? '' : 'dark');
    document.getElementById('themeBtn').textContent = isDark ? '🌙 Dark mode' : '☀️ Light mode';

    // Update chart colours to match the new theme
    const tickColor  = isDark ? '#64748b' : '#94a3b8';
    const gridColor  = isDark ? '#e2e8f0' : '#334155';
    const labelColor = isDark ? '#0f172a' : '#f1f5f9';

    [anomalyChartObj, wordFreqChartObj].forEach(chart => {{
      chart.options.scales.x.ticks.color = tickColor;
      chart.options.scales.x.grid.color  = gridColor;
      chart.options.scales.y.ticks.color = tickColor;
      chart.options.scales.y.grid.color  = gridColor;
      chart.options.plugins.legend.labels.color = labelColor;
      chart.update();
    }});
  }}
</script>
</body>
</html>"""

with open("dashboard.html", "w") as f:
    f.write(html)

print("Dashboard written to dashboard.html")
