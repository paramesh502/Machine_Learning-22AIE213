import { Marker } from "./api";

export function cn(...classes: (string | false | null | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}

/**
 * Highlight low/high markers in the original text, returning an array
 * of React-friendly spans.  Matches are case-insensitive but positions
 * come from the lower-cased scan on the backend so we use the raw
 * `text` for display and pick positions from there.
 */
export type Segment =
  | { kind: "plain"; value: string }
  | { kind: "low"; value: string }
  | { kind: "high"; value: string };

export function annotate(
  text: string,
  low: Marker[],
  high: Marker[]
): Segment[] {
  type Span = { kind: "low" | "high"; start: number; end: number };
  const spans: Span[] = [
    ...low.map((m) => ({ kind: "low" as const, start: m.start, end: m.end })),
    ...high.map((m) => ({ kind: "high" as const, start: m.start, end: m.end })),
  ].sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;
  for (const s of spans) {
    if (s.start < cursor) continue; // skip overlaps
    if (s.start > cursor) {
      segments.push({ kind: "plain", value: text.slice(cursor, s.start) });
    }
    segments.push({ kind: s.kind, value: text.slice(s.start, s.end) });
    cursor = s.end;
  }
  if (cursor < text.length) {
    segments.push({ kind: "plain", value: text.slice(cursor) });
  }
  return segments;
}

export function levelToColor(level: string) {
  switch (level) {
    case "High":
      return "#34c759";
    case "Medium":
      return "#ff9f0a";
    case "Low":
      return "#ff3b30";
    default:
      return "#0071e3";
  }
}

export function formatDate(ts: number) {
  return new Date(ts).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
