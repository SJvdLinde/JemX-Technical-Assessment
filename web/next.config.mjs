/** @type {import('next').NextConfig} */
export default {
  // The API base URL is the only thing that changes between local and deployed.
  env: { NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000" },
};
