import { RemoteZipPointer } from '@gyng/remote-zip';
import fs from 'node:fs/promises';
import path from 'node:path';

const archiveUrl = new URL('https://zenodo.org/records/7260705/files/Task1.zip?download=1');
const archiveBytes = 14_471_900_926;
const center = String(process.argv[2] || '').toUpperCase();
const start = Number(process.argv[3] || 0);
const count = Number(process.argv[4] || 6);
const outRoot = process.argv[5] || `data/full/${center}_${start}`;
const maxAttempts = 8;

if (!['A', 'B', 'C'].includes(center)) throw new Error('Center must be A, B, or C');
if (!Number.isInteger(start) || start < 0 || !Number.isInteger(count) || count < 1) throw new Error('Invalid start/count');
await fs.mkdir(outRoot, { recursive: true });

const nativeFetch = globalThis.fetch;
globalThis.fetch = async (input, init = {}) => {
  const method = String(init?.method || 'GET').toUpperCase();
  const rawUrl = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
  const url = new URL(rawUrl);
  if (method === 'HEAD' && url.hostname.endsWith('zenodo.org')) {
    return new Response(null, { status: 200, headers: { 'content-length': String(archiveBytes) } });
  }
  return nativeFetch(input, init);
};

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function openArchive() {
  return new RemoteZipPointer({
    url: archiveUrl,
    timeoutMs: 300000,
    requestInit: {
      cache: 'no-store',
      headers: {
        'User-Agent': 'FedSynthCT-Research-Reanalysis/1.0',
        Accept: 'application/octet-stream,*/*',
        Connection: 'keep-alive',
      },
    },
  }).populate();
}

async function fetchMemberWithRetry(member) {
  let lastError;
  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      console.log(`Fetching ${member}; attempt ${attempt}/${maxAttempts}`);
      const activeZip = await openArchive();
      const bytes = await activeZip.fetch(member, undefined, {
        maxUncompressedSize: 1024 * 1024 * 1024,
        verifyCrc: true,
        timeoutMs: 600000,
      });
      if (!bytes || bytes.byteLength === 0) throw new Error(`Empty archive member: ${member}`);
      return bytes;
    } catch (error) {
      lastError = error;
      const waitMs = Math.min(60_000, 3_000 * (2 ** (attempt - 1))) + Math.floor(Math.random() * 2_000);
      console.warn(`Attempt ${attempt} failed for ${member}: ${error?.message || error}`);
      if (attempt < maxAttempts) {
        console.warn(`Retrying after ${waitMs} ms`);
        await sleep(waitMs);
      }
    }
  }
  throw new Error(`Failed to fetch ${member} after ${maxAttempts} attempts: ${lastError?.message || lastError}`);
}

const zip = await openArchive();
const entries = zip.files().map((entry) => entry.filename).sort();
const pattern = new RegExp(`(?:^|/)(Task1/brain/(1B${center}\\d{3}))/mr\\.nii\\.gz$`);
const mapping = new Map();
for (const filename of entries) {
  const match = filename.match(pattern);
  if (match) mapping.set(match[2], match[1]);
}

const allCases = [...mapping.keys()].sort();
if (allCases.length !== 60) console.warn(`Expected 60 center-${center} cases; found ${allCases.length}`);
const selected = allCases.slice(start, start + count);
if (selected.length === 0) throw new Error(`No cases selected for ${center} at start ${start}`);
console.log(`Center ${center}; selected ${selected.join(', ')}`);

const manifest = [];
for (const caseId of selected) {
  const caseDir = path.join(outRoot, caseId);
  await fs.mkdir(caseDir, { recursive: true });
  for (const filename of ['mr.nii.gz', 'ct.nii.gz', 'mask.nii.gz']) {
    const member = `${mapping.get(caseId)}/${filename}`;
    const bytes = await fetchMemberWithRetry(member);
    const target = path.join(caseDir, filename);
    await fs.writeFile(target, bytes);
    manifest.push({ center, case_id: caseId, member, target, bytes: bytes.byteLength });
  }
}

await fs.writeFile(
  path.join(outRoot, 'extraction_manifest.json'),
  JSON.stringify({
    archive: archiveUrl.toString(),
    archive_bytes: archiveBytes,
    center,
    start,
    count,
    selected,
    max_attempts_per_member: maxAttempts,
    manifest,
  }, null, 2),
);
