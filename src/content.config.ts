import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Blog system — "Notes". Markdown/MDX posts with technical-blog frontmatter.
const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    author: z.string().default('Youngmin Cho'),
    tags: z.array(z.string()).default([]),
    subtitle: z.string().optional(),
    draft: z.boolean().default(false),
  }),
});

// Docs — skeleton only for now (nav structure; prose added later).
const docs = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/docs' }),
  schema: z.object({
    title: z.string(),
    order: z.number().default(0),
    summary: z.string().optional(),
  }),
});

export const collections = { blog, docs };
