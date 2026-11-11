<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import bookManifest from '../../../book.manifest.json'

type Locale = 'en' | 'ja'
type LocalizedTitle = Record<Locale, string>
type BookChapter = { route: string; title: LocalizedTitle }
type BookPart = {
  number: number
  dir: string
  title: LocalizedTitle
  chapters: BookChapter[]
}

const parts = (bookManifest as unknown as { parts: BookPart[] }).parts

/**
 * The running head is resolved from the manifest, the same source as the sidebar and the
 * print order, so a chapter can never announce a part or position it does not have.
 */
const positions = new Map<string, { part: BookPart; index: number }>()
for (const part of parts) {
  part.chapters.forEach((chapter, index) => positions.set(chapter.route, { part, index }))
}

const { page } = useData()

const locale = computed<Locale>(() => (page.value.relativePath.startsWith('ja/') ? 'ja' : 'en'))

const entry = computed(() => {
  const route = page.value.relativePath.replace(/^ja\//, '').replace(/\.md$/, '')
  const found = positions.get(route)
  if (!found) return null

  const { part, index } = found
  const ja = locale.value === 'ja'
  return {
    partHref: withBase(`/${ja ? 'ja/' : ''}${part.chapters[0].route}`),
    partLabel: ja ? `第${part.number}部` : `Part ${part.number}`,
    partTitle: part.title[locale.value],
    folio: ja
      ? `第${index + 1}章 / 全${part.chapters.length}章`
      : `Chapter ${index + 1} of ${part.chapters.length}`,
  }
})
</script>

<template>
  <div v-if="entry" class="chapter-opener" :lang="locale">
    <a class="opener-part" :href="entry.partHref">
      <b>{{ entry.partLabel }}</b><i aria-hidden="true">·</i>{{ entry.partTitle }}
    </a>
    <span class="opener-folio">{{ entry.folio }}</span>
  </div>
</template>
