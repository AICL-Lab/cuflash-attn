#!/usr/bin/env node
/**
 * Sync CHANGELOG.md to docs/en/release-notes/changelog.md and docs/zh/release-notes/changelog.md
 *
 * This script copies the content from the root CHANGELOG.md to the docs site,
 * with only formatting changes (title format).
 *
 * Run from the docs directory: node scripts/sync-changelog.mjs
 */

import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const docsDir = join(__dirname, "..");
const rootDir = join(docsDir, "..");

const sourcePath = join(rootDir, "CHANGELOG.md");

const enTargetPath = join(docsDir, "en/release-notes/changelog.md");
const zhTargetPath = join(docsDir, "zh/release-notes/changelog.md");

const EN_HEADER = `# Changelog

This page documents the changes in each CuFlash-Attn release.

`;

const ZH_HEADER = `# 更新日志

本页记录 CuFlash-Attn 每个版本的变更内容。

`;

// Read the source file
let content = readFileSync(sourcePath, "utf-8");

// Remove the HTML comment block at the top
content = content.replace(/<!--[\s\S]*?-->\n*/g, "");

// Remove the "# Changelog" title (we'll add our own header)
content = content.replace(/^# Changelog\n+/, "");

// Convert title format: ## [0.3.0] - 2026-04-24 -> ## 0.3.0 (2026-04-24)
content = content.replace(
  /^## \[([^\]]+)\] - (\d{4}-\d{1,2}-\d{1,2})/gm,
  "## $1 ($2)"
);

// Replace old repository URLs if present (for repos that migrated)
content = content.replace(/LessUp\/cuflash-attn/g, "AICL-Lab/cuflash-attn");

// Ensure target directories exist
for (const targetPath of [enTargetPath, zhTargetPath]) {
  mkdirSync(dirname(targetPath), { recursive: true });
}

// Write the target files
writeFileSync(enTargetPath, EN_HEADER + content.trim() + "\n");
writeFileSync(zhTargetPath, ZH_HEADER + content.trim() + "\n");

console.log(`Synced changelog to ${enTargetPath}`);
console.log(`Synced changelog to ${zhTargetPath}`);
