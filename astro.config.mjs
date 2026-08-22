import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';

// GitHub Pages project site: https://youngmin17.github.io/Youngmin17
// (profile README still renders on the GitHub profile page; this is the standalone site.)
export default defineConfig({
  site: 'https://youngmin17.github.io',
  base: '/Youngmin17',
  integrations: [mdx(), react()],
});
