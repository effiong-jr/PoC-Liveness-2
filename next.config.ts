import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Serve WASM files with the correct MIME type
  async headers() {
    return [
      {
        source: "/:path*.wasm",
        headers: [{ key: "Content-Type", value: "application/wasm" }],
      },
    ];
  },
  // Turbopack (default in Next.js 16) — empty object silences the webpack
  // conflict warning. Turbopack handles WASM natively and automatically
  // excludes Node.js built-ins from browser bundles.
  turbopack: {},
};

export default nextConfig;
