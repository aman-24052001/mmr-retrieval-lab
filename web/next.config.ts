import type { NextConfig } from "next";
const nextConfig: NextConfig = {
  output: "export",
  basePath: "/mmr-retrieval-lab",
  assetPrefix: "/mmr-retrieval-lab/",
  images: { unoptimized: true },
};
export default nextConfig;
