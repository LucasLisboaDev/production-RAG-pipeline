/**
 * ArXiv RAG Chat UI
 * -----------------
 * Single-page React chat that talks to the FastAPI backend at POST /query.
 * Sends the user's question, shows a typing indicator while waiting, then
 * renders the grounded answer and clickable ArXiv source citations.
 *
 * Configure the API base URL with VITE_API_URL (defaults to http://localhost:8000).
 */

import { useState, useEffect, useRef } from "react";

const API_URL =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

function yearFromPublished(published) {
  if (!published) return "";
  const y = String(published).slice(0, 4);
  return /^\d{4}$/.test(y) ? y : published;
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-4 py-3">
      <span
        className="h-2 w-2 rounded-full bg-gray-400 animate-bounce"
        style={{ animationDelay: "0ms" }}
      />
      <span
        className="h-2 w-2 rounded-full bg-gray-400 animate-bounce"
        style={{ animationDelay: "150ms" }}
      />
      <span
        className="h-2 w-2 rounded-full bg-gray-400 animate-bounce"
        style={{ animationDelay: "300ms" }}
      />
    </div>
  );
}

function Sources({ citations }) {
  if (!citations?.length) return null;

  return (
    <div className="mt-3 border-t border-gray-200 pt-3">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
        Sources
      </p>
      <ul className="space-y-1.5">
        {citations.map((c) => (
          <li key={`${c.index}-${c.arxiv_url}`}>
            <a
              href={c.arxiv_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
            >
              [{c.index}] {c.title} ({yearFromPublished(c.published)})
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-3 text-white shadow-sm">
          <p className="whitespace-pre-wrap text-sm leading-relaxed">
            {message.content}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-2xl rounded-bl-md bg-gray-100 px-4 py-3 text-gray-900 shadow-sm">
        {message.loading ? (
          <TypingIndicator />
        ) : (
          <>
            <p className="whitespace-pre-wrap text-sm leading-relaxed">
              {message.content}
            </p>
            <Sources citations={message.citations} />
          </>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const handleNewChat = () => {
    setMessages([]);
    setInput("");
    setIsLoading(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const question = input.trim();
    if (!question || isLoading) return;

    setInput("");
    setIsLoading(true);

    const userMessage = { id: crypto.randomUUID(), role: "user", content: question };
    const assistantId = crypto.randomUUID();
    const loadingMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      loading: true,
      citations: [],
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);

    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();

      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                loading: false,
                content: data.answer,
                citations: data.citations ?? [],
              }
            : m
        )
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                loading: false,
                content: `Sorry, something went wrong: ${err.message}`,
                citations: [],
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-white text-gray-900">
      {/* Sidebar */}
      <aside className="flex w-64 shrink-0 flex-col border-r border-gray-800 bg-gray-900 text-gray-100">
        <div className="border-b border-gray-800 px-4 py-5">
          <h1 className="text-lg font-semibold tracking-tight">ArXiv RAG</h1>
          <p className="mt-1 text-xs text-gray-400">Ask My ArXiv Docs</p>
        </div>
        <div className="p-3">
          <button
            type="button"
            onClick={handleNewChat}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-sm font-medium text-gray-100 transition hover:bg-gray-700"
          >
            New Chat
          </button>
        </div>
      </aside>

      {/* Main chat */}
      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-center text-gray-500">
              <p className="text-lg font-medium text-gray-700">
                Ask about AI &amp; ML research
              </p>
              <p className="mt-2 max-w-md text-sm">
                Try: &quot;What are the limitations of retrieval augmented
                generation systems?&quot;
              </p>
            </div>
          ) : (
            <div className="mx-auto flex max-w-3xl flex-col gap-6">
              {messages.map((msg) => (
                <MessageBubble key={msg.id} message={msg} />
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <form
          onSubmit={handleSubmit}
          className="border-t border-gray-200 bg-white px-4 py-4 md:px-8"
        >
          <div className="mx-auto flex max-w-3xl gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about ArXiv papers…"
              disabled={isLoading}
              className="flex-1 rounded-xl border border-gray-300 px-4 py-3 text-sm outline-none transition focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 disabled:bg-gray-50"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              Send
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
