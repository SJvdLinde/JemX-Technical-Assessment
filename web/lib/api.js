export const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function unwrap(response) {
  if (!response.ok) {
    // The backend returns a readable sentence in `detail` for every bad upload.
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

export async function fetchAnalysis() {
  return unwrap(await fetch(`${API}/api/analyse`, { cache: "no-store" }));
}

export async function uploadExport(fileList) {
  const form = new FormData();
  for (const file of fileList) form.append("files", file);
  return unwrap(await fetch(`${API}/api/analyse`, { method: "POST", body: form }));
}

export const rand = (n) =>
  n == null ? "—" : "R" + Math.round(n).toLocaleString("en-ZA");
export const hrs = (n) => (n == null ? "—" : `${n.toFixed(1)}h`);
