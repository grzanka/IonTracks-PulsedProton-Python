// The whole app runs client-side (WebAssembly + a Web Worker) and has no
// server data to render, so skip SSR entirely and prerender the static shell.
export const ssr = false;
export const prerender = true;
