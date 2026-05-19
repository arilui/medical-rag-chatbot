"use client";

import { useState } from "react";

type Source = {
  source: string;
  page?: number;
  preview: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!input.trim()) return;

    const userMessage: Message = {
      role: "user",
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);

    const currentInput = input;
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: currentInput,
        }),
      });

      const data = await response.json();

      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Error connecting to backend.",
        },
      ]);
    }

    setLoading(false);
  }

  return (
    <main className="flex flex-col h-screen bg-zinc-950 text-white">
      {/* Header */}
      <div className="border-b border-zinc-800 p-4">
        <h1 className="text-2xl font-bold">
          Medical RAG Assistant
        </h1>

        <p className="text-zinc-400 text-sm mt-1">
          Educational medical information assistant powered by RAG
        </p>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`max-w-3xl ${
              message.role === "user"
                ? "ml-auto"
                : "mr-auto"
            }`}
          >
            <div
              className={`rounded-2xl p-4 whitespace-pre-wrap ${
                message.role === "user"
                  ? "bg-blue-600"
                  : "bg-zinc-800"
              }`}
            >
              {message.content}
            </div>

            {/* Sources */}
            {message.sources && (
              <div className="mt-3 space-y-2">
                {message.sources.map((source, idx) => (
                  <div
                    key={idx}
                    className="bg-zinc-900 border border-zinc-800 rounded-xl p-3 text-sm"
                  >
                    <div className="font-semibold">
                      {source.source}
                    </div>

                    <div className="text-zinc-400 mt-1">
                      {source.preview}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="bg-zinc-800 rounded-2xl p-4 w-fit">
            Thinking...
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-zinc-800 p-4">
        <div className="max-w-4xl mx-auto flex gap-3">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                sendMessage();
              }
            }}
            placeholder="Ask a medical question..."
            className="flex-1 bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3 outline-none focus:border-blue-500"
          />

          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-xl font-semibold"
          >
            Send
          </button>
        </div>
      </div>
    </main>
  );
}