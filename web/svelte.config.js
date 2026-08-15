import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

// GitHub Pages serves project sites under a sub-path (e.g. /IonTracks-PulsedProton-Python).
// CI's deploy workflow sets BASE_PATH for the deploy build; local dev leaves it empty so the
// app is served from the origin root. SvelteKit requires base to start with a leading slash and
// have no trailing slash, so normalize both: strip trailing slashes and force a single leading
// slash even if BASE_PATH omits it.
const rawBasePath = process.env.BASE_PATH?.trim().replace(/\/+$/, "");
const basePath = rawBasePath && rawBasePath !== "/" ? `/${rawBasePath.replace(/^\/+/, "")}` : "";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    // Prerendered HTML + SPA fallback: the whole app runs client-side (WASM +
    // a Web Worker), so there is nothing for a server to render.
    adapter: adapter({
      fallback: "404.html",
    }),
    prerender: {
      handleHttpError: "warn",
    },
    paths: {
      base: basePath,
    },
    alias: {
      $lib: "./src/lib",
    },
  },
};

export default config;
