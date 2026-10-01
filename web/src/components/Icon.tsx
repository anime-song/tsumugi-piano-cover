// 線のアイコン (24px の viewBox、色は文字色)
const PATHS: Record<string, string> = {
  play: "M7 4.5v15l12.5-7.5z",
  pause: "M7 4.5h3.5v15H7zM13.5 4.5H17v15h-3.5z",
  back: "M11 6 5 12l6 6M19 6l-6 6 6 6",
  forward: "M13 6l6 6-6 6M5 6l6 6-6 6",
  loop: "M4 12a6 6 0 0 1 6-6h8m-3-3 3 3-3 3M20 12a6 6 0 0 1-6 6H6m3 3-3-3 3-3",
  star: "M12 3.5l2.6 5.3 5.9.9-4.25 4.1 1 5.8L12 16.9l-5.25 2.75 1-5.8L3.5 9.7l5.9-.9z",
  download: "M12 4v11m-5-5 5 5 5-5M5 20h14",
  trash: "M5 7h14M10 7V4.5h4V7M7 7l1 13h8l1-13",
  reuse: "M4 4v6h6M5 15a7.5 7.5 0 1 0 1.5-7.5L4 10",
  scissors: "M12 3v18M7 8l5-5 5 5",
  stop: "M7 7h10v10H7z",
  plus: "M12 5v14M5 12h14",
  close: "M6 6l12 12M18 6 6 18",
  music: "M9 18V6l10-2v12M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0zM19 16a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z",
  sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8zM19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z",
  layers: "M12 4 3 9l9 5 9-5zM3 14l9 5 9-5",
  chip: "M7 7h10v10H7zM9.5 3v4M14.5 3v4M9.5 17v4M14.5 17v4M3 9.5h4M3 14.5h4M17 9.5h4M17 14.5h4",
};

export function Icon({ name, size = 18, filled }: { name: string; size?: number; filled?: boolean }) {
  const solid = filled ?? ["play", "pause", "stop"].includes(name);
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={solid ? "currentColor" : "none"}
      stroke={solid ? "none" : "currentColor"}
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[name] ?? ""} />
    </svg>
  );
}
