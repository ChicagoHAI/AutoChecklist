"use client";

import { useRef, useCallback } from "react";
import Editor, { OnMount } from "@monaco-editor/react";
import type { editor, IPosition } from "monaco-editor";

interface PromptEditorProps {
  value: string;
  onChange: (value: string) => void;
  readOnly?: boolean;
  minHeight?: number;
}

const PLACEHOLDER_COLORS: Record<string, string> = {
  input: "#1a7f37",
  target: "#0969da",
  reference: "#8250df",
  candidates: "#cf222e",
  checklist: "#8250df",
};
const DEFAULT_PLACEHOLDER_COLOR = "#9a6700";

export function PromptEditor({
  value,
  onChange,
  readOnly = false,
  minHeight = 200,
}: PromptEditorProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);
  const decorationsRef = useRef<editor.IEditorDecorationsCollection | null>(null);

  const updateDecorations = useCallback(() => {
    const ed = editorRef.current;
    if (!ed) return;

    const model = ed.getModel();
    if (!model) return;

    const decorations: editor.IModelDeltaDecoration[] = [];
    const text = model.getValue();
    const placeholderRegex = /\{(\w+)\}/g;
    let match;

    const HOVER_MESSAGES: Record<string, string> = {
      input: "**{input}** — The instruction or query being evaluated",
      target: "**{target}** — The response to evaluate",
      reference: "**{reference}** — Optional gold reference target",
      candidates: "**{candidates}** — Auto-generated alternative responses",
      checklist: "**{checklist}** — Checklist questions to score against",
    };

    while ((match = placeholderRegex.exec(text)) !== null) {
      const startPos = model.getPositionAt(match.index);
      const endPos = model.getPositionAt(match.index + match[0].length);
      const placeholder = match[1];
      decorations.push({
        range: {
          startLineNumber: startPos.lineNumber,
          startColumn: startPos.column,
          endLineNumber: endPos.lineNumber,
          endColumn: endPos.column,
        },
        options: {
          inlineClassName: `placeholder-${placeholder in PLACEHOLDER_COLORS ? placeholder : "other"}`,
          hoverMessage: {
            value: HOVER_MESSAGES[placeholder] || `**{${placeholder}}** — Placeholder`,
          },
        },
      });
    }

    if (decorationsRef.current) {
      decorationsRef.current.clear();
    }
    decorationsRef.current = ed.createDecorationsCollection(decorations);
  }, []);

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;

    // Define a grayed-out theme for read-only state
    monaco.editor.defineTheme("vs-readonly", {
      base: "vs",
      inherit: true,
      rules: [],
      colors: {
        "editor.background": "#f5f5f5",
        "editor.foreground": "#666666",
      },
    });

    // Add CSS for placeholder highlighting
    const styleEl = document.getElementById("prompt-editor-styles") || document.createElement("style");
    styleEl.id = "prompt-editor-styles";
    styleEl.textContent = [
      ...Object.entries(PLACEHOLDER_COLORS).map(
        ([name, color]) =>
          `.placeholder-${name} { color: ${color} !important; font-weight: 600; background: ${color}15; border-radius: 2px; }`
      ),
      `.placeholder-other { color: ${DEFAULT_PLACEHOLDER_COLOR} !important; font-weight: 600; background: ${DEFAULT_PLACEHOLDER_COLOR}15; border-radius: 2px; }`,
    ].join("\n");
    if (!document.getElementById("prompt-editor-styles")) {
      document.head.appendChild(styleEl);
    }

    // Register completion provider for placeholders
    monaco.languages.registerCompletionItemProvider("markdown", {
      triggerCharacters: ["{"],
      provideCompletionItems: (model: editor.ITextModel, position: IPosition) => {
        const word = model.getWordUntilPosition(position);
        const range = {
          startLineNumber: position.lineNumber,
          startColumn: word.startColumn,
          endLineNumber: position.lineNumber,
          endColumn: word.endColumn,
        };

        return {
          suggestions: [
            {
              label: "{input}",
              kind: monaco.languages.CompletionItemKind.Variable,
              insertText: "input}",
              range,
              detail: "The instruction or query",
            },
            {
              label: "{target}",
              kind: monaco.languages.CompletionItemKind.Variable,
              insertText: "target}",
              range,
              detail: "The response to evaluate",
            },
            {
              label: "{reference}",
              kind: monaco.languages.CompletionItemKind.Variable,
              insertText: "reference}",
              range,
              detail: "Optional gold reference",
            },
            {
              label: "{candidates}",
              kind: monaco.languages.CompletionItemKind.Variable,
              insertText: "candidates}",
              range,
              detail: "Auto-generated alternative responses",
            },
            {
              label: "{checklist}",
              kind: monaco.languages.CompletionItemKind.Variable,
              insertText: "checklist}",
              range,
              detail: "Checklist questions (scorer)",
            },
          ],
        };
      },
    });

    updateDecorations();
    editor.onDidChangeModelContent(() => updateDecorations());
  };

  return (
    <div
      className="rounded-md overflow-hidden"
      style={{
        border: "1px solid var(--border)",
        minHeight,
      }}
    >
      <Editor
        height={`${minHeight}px`}
        defaultLanguage="markdown"
        value={value}
        onChange={(val) => onChange(val || "")}
        onMount={handleMount}
        options={{
          readOnly,
          readOnlyMessage: { value: "Clear the result to edit the prompt." },
          minimap: { enabled: false },
          lineNumbers: "off",
          glyphMargin: false,
          folding: false,
          wordWrap: "on",
          wrappingStrategy: "advanced",
          scrollBeyondLastLine: false,
          automaticLayout: true,
          fontSize: 13,
          fontFamily: "var(--font-mono), monospace",
          padding: { top: 12, bottom: 12 },
          renderLineHighlight: "none",
          overviewRulerLanes: 0,
          hideCursorInOverviewRuler: true,
          scrollbar: {
            vertical: "auto",
            horizontal: "hidden",
            verticalScrollbarSize: 8,
          },
        }}
        theme={readOnly ? "vs-readonly" : "vs"}
      />
    </div>
  );
}
