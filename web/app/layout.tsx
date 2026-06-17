import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MMR Retrieval Lab — diversity vs relevance",
  description: "Interactive Maximal Marginal Relevance demo. Drag lambda, watch the selected set change.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
