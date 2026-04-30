import { useState, useRef, useEffect } from "react";
import "./App.css";

const API = "http://localhost:8000";

const SUGGESTIONS = [
  "What are the main topics covered in this document?",
  "Summarize the key findings from the first section.",
  "What methodology is described in the paper?",
  "What conclusions does the document draw?",
];

const DEFAULT_SETTINGS = {
  topK: 5,
  maxTokens: 512,
  temperature: 0.70,
  darkMode: false,
};

// ── icons ────────────────────────────────────────────────────────────────────

function BotIcon({ size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
      <rect x="3" y="8" width="18" height="13" rx="3" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 8V6a4 4 0 0 1 8 0v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <circle cx="9" cy="14" r="1.2" fill="currentColor" />
      <circle cx="15" cy="14" r="1.2" fill="currentColor" />
      <path d="M9.5 17.5h5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
      <path d="M12 20V4M5 11l7-7 7 7" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <rect x="4" y="4" width="16" height="16" rx="2" />
    </svg>
  );
}

function DocIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function NewChatIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
      <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.6" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
      <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="5" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 2v2M12 20v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2 12h2M20 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── toggle switch ─────────────────────────────────────────────────────────────

function Toggle({ checked, onChange }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="toggle-track">
        <span className="toggle-thumb" />
      </span>
    </label>
  );
}

// ── slider row ────────────────────────────────────────────────────────────────

function SliderRow({ label, value, min, max, step, display, onChange }) {
  return (
    <div className="setting-item">
      <div className="setting-item-header">
        <span className="setting-label">{label}</span>
        <span className="setting-value">{display ?? value}</span>
      </div>
      <input
        type="range"
        className="slider"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="slider-bounds">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

// ── settings panel ────────────────────────────────────────────────────────────

function SettingsPanel({ settings, update, onClose }) {
  return (
    <aside className="settings-panel">
      <div className="settings-header">
        <span className="settings-title">Settings</span>
        <button className="settings-close" onClick={onClose} aria-label="Close settings">
          <XIcon />
        </button>
      </div>

      <div className="settings-body">
        {/* appearance */}
        <section className="settings-section">
          <p className="settings-section-label">Appearance</p>

          <div className="setting-item setting-item-row">
            <div className="setting-item-left">
              <span className="setting-icon">
                {settings.darkMode ? <MoonIcon /> : <SunIcon />}
              </span>
              <span className="setting-label">{settings.darkMode ? "Dark mode" : "Light mode"}</span>
            </div>
            <Toggle checked={settings.darkMode} onChange={(v) => update("darkMode", v)} />
          </div>
        </section>

        <div className="settings-divider" />

        {/* retrieval */}
        <section className="settings-section">
          <p className="settings-section-label">Retrieval</p>

          <SliderRow
            label="Top K chunks"
            value={settings.topK}
            min={1}
            max={15}
            step={1}
            onChange={(v) => update("topK", v)}
          />

          <div className="setting-desc">
            How many document chunks are retrieved and fed to the model as context.
          </div>
        </section>

        <div className="settings-divider" />

        {/* generation */}
        <section className="settings-section">
          <p className="settings-section-label">Generation</p>

          <SliderRow
            label="Max tokens"
            value={settings.maxTokens}
            min={64}
            max={2048}
            step={64}
            onChange={(v) => update("maxTokens", v)}
          />

          <SliderRow
            label="Temperature"
            value={settings.temperature}
            min={0}
            max={1.5}
            step={0.05}
            display={settings.temperature.toFixed(2)}
            onChange={(v) => update("temperature", v)}
          />

          <div className="setting-desc">
            Higher temperature = more creative but less precise answers.
          </div>
        </section>

        <div className="settings-divider" />

        {/* model */}
        <section className="settings-section">
          <p className="settings-section-label">Model</p>

          <div className="model-card">
            <div className="model-card-icon">
              <BotIcon size={18} />
            </div>
            <div className="model-card-info">
              <p className="model-card-name">Llama 3.1 8B Instruct</p>
              <p className="model-card-file">Q4_0 · 128k context · GPT4All</p>
            </div>
            <span className="model-card-badge">Active</span>
          </div>

          <div className="setting-desc">
            Model changes require a server restart.
          </div>
        </section>

        <div className="settings-divider" />

        {/* reset */}
        <section className="settings-section">
          <button
            className="reset-btn"
            onClick={() => {
              update("topK", DEFAULT_SETTINGS.topK);
              update("maxTokens", DEFAULT_SETTINGS.maxTokens);
              update("temperature", DEFAULT_SETTINGS.temperature);
            }}
          >
            Reset to defaults
          </button>
        </section>
      </div>
    </aside>
  );
}

// ── source card ───────────────────────────────────────────────────────────────

function SourceCard({ chunk }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`source-card ${open ? "open" : ""}`}>
      <button className="source-header" onClick={() => setOpen(!open)}>
        <span className="source-icon"><DocIcon /></span>
        <span className="source-title">Chunk {chunk.chunk_id}</span>
        <span className="source-score">{(chunk.score * 100).toFixed(1)}%</span>
        <span className={`source-chevron ${open ? "flipped" : ""}`}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
            <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>
      {open && <p className="source-text">{chunk.text}</p>}
    </div>
  );
}

// ── message ───────────────────────────────────────────────────────────────────

function Message({ msg }) {
  if (msg.role === "user") {
    return (
      <div className="message-row user-row">
        <div className="bubble user">{msg.content}</div>
        <div className="avatar user-avatar"><span>U</span></div>
      </div>
    );
  }
  if (msg.role === "error") {
    return (
      <div className="message-row">
        <div className="avatar bot-avatar"><BotIcon /></div>
        <div className="bubble error-bubble">{msg.content}</div>
      </div>
    );
  }
  return (
    <div className="assistant-block">
      <div className="message-row">
        <div className="avatar bot-avatar"><BotIcon /></div>
        <div className="bubble assistant">{msg.content}</div>
      </div>
      {msg.sources?.length > 0 && (
        <div className="sources">
          <p className="sources-label">Sources</p>
          <div className="source-list">
            {msg.sources.map((s) => (
              <SourceCard key={s.chunk_id} chunk={s} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── app ───────────────────────────────────────────────────────────────────────

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);

  const bottomRef = useRef(null);
  const textareaRef = useRef(null);
  const abortRef = useRef(null);

  function update(key, value) {
    setSettings((s) => ({ ...s, [key]: value }));
  }

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", settings.darkMode ? "dark" : "light");
  }, [settings.darkMode]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
  }, [input]);

  async function send() {
    const question = input.trim();
    if (!question || loading) return;

    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`${API}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          top_k: settings.topK,
          max_tokens: settings.maxTokens,
          temperature: settings.temperature,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        let msg = "Server error";
        try { msg = (await res.json()).detail || msg; } catch { /* non-JSON body */ }
        throw new Error(msg);
      }

      setMessages((m) => [...m, { role: "assistant", content: "", sources: [] }]);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let event;
          try { event = JSON.parse(line.slice(6)); } catch { continue; }

          if (event.type === "sources") {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1] = { ...copy[copy.length - 1], sources: event.sources };
              return copy;
            });
          } else if (event.type === "token") {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1] = {
                ...copy[copy.length - 1],
                content: copy[copy.length - 1].content + event.token,
              };
              return copy;
            });
          }
        }
      }
    } catch (e) {
      if (e.name !== "AbortError") {
        setMessages((m) => [...m, { role: "error", content: `Error: ${e.message}` }]);
      }
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function newChat() {
    stop();
    setMessages([]);
    setInput("");
  }

  function onKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  function applySuggestion(text) {
    setInput(text);
    textareaRef.current?.focus();
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">
            <BotIcon size={18} />
          </div>
          <div>
            <h1 className="header-title">
              RAG<span className="gradient-text">Chat</span>
            </h1>
            <p className="header-sub">50-pages.pdf · Llama 3.1 8B</p>
          </div>
        </div>

        <div className="header-actions">
          <div className="header-badge">Local AI</div>
          <button
            className="gear-btn new-chat-btn"
            onClick={newChat}
            aria-label="New chat"
            title="New chat"
          >
            <NewChatIcon />
            <span>New Chat</span>
          </button>
          <button
            className={`gear-btn ${settingsOpen ? "active" : ""}`}
            onClick={() => setSettingsOpen((o) => !o)}
            aria-label="Toggle settings"
          >
            <GearIcon />
          </button>
        </div>
      </header>

      <div className="layout">
        <div className="chat-pane">
          <main className="chat">
            {messages.length === 0 && (
              <div className="empty">
                <div className="empty-icon">
                  <BotIcon size={26} />
                </div>
                <h2 className="empty-title">Ask about your document</h2>
                <p className="empty-sub">I've read all 50 pages. Ask me anything.</p>
                <div className="suggestions">
                  {SUGGESTIONS.map((s, i) => (
                    <button key={i} className="suggestion-btn" onClick={() => applySuggestion(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg, i) => (
              <Message key={i} msg={msg} />
            ))}
            {loading && (
              <div className="message-row">
                <div className="avatar bot-avatar"><BotIcon /></div>
                <div className="bubble assistant loading-bubble">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </main>

          <footer className="input-area">
            <div className="input-wrapper">
              <textarea
                ref={textareaRef}
                className="input"
                rows={1}
                placeholder="Ask anything… (Shift+Enter for new line)"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={onKey}
                disabled={loading}
              />
              {loading ? (
                <button className="stop-btn" onClick={stop} aria-label="Stop generating">
                  <StopIcon />
                </button>
              ) : (
                <button
                  className="send-btn"
                  onClick={send}
                  disabled={!input.trim()}
                  aria-label="Send"
                >
                  <SendIcon />
                </button>
              )}
            </div>
            <p className="input-hint">
              {loading ? "Generating… click ■ to stop" : "Enter to send · Shift+Enter for new line"}
            </p>
          </footer>
        </div>

        {settingsOpen && (
          <SettingsPanel
            settings={settings}
            update={update}
            onClose={() => setSettingsOpen(false)}
          />
        )}
      </div>
    </div>
  );
}
