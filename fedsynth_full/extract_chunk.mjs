import { RemoteZipPointer } from '@gyng/remote-zip';
import fs from 'node:fs/promises';
import path from 'node:path';

const archiveUrl = new URL('https://zenodo.org/records/7260705/files/Task1.zip?download=1');
const archiveBytes = 14_471_900_926;
const center = String(process.argv[2] || '').toUpperCase();
const start = Number(process.argv[3] || 0);
const count = Number(process.argv[4] || 6);
const outRoot = process.argv[5] || `data/full/${center}_${start}`;
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

const zip = await new RemoteZipPointer({
  url: archiveUrl,
  timeoutMs: 180000,
  requestInit: { cache: 'no-store', headers: { 'User-Agent': 'FedSynthCT-Research-Reanalysis/1.0', Accept: 'application/octet-stream,*/*' } },
}).populate();
const entries = zip.files().map((e) => e.filename).sort();
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
    const bytes = await zip.fetch(member, undefined, { maxUncompressedSize: 1024 * 1024 * 1024, verifyCrc: true, timeoutMs: 300000 });
    const target = path.join(caseDir, filename);
    await fs.writeFile(target, bytes);
    manifest.push({ center, case_id: caseId, member, target, bytes: bytes.byteLength });
  }
}
await fs.writeFile(path.join(outRoot, 'extraction_manifest.json'), JSON.stringify({ archive: archiveUrl.toString(), center, start, count, selected, manifest }, null, 2));
