import { RemoteZipPointer } from '@gyng/remote-zip';
import fs from 'node:fs/promises';
import path from 'node:path';

const archiveUrl = new URL('https://zenodo.org/records/7260705/files/Task1.zip?download=1');
const archiveBytes = 14_471_900_926;
const outRoot = process.argv[2] || 'data/pilot';
await fs.mkdir(outRoot, { recursive: true });

// Zenodo serves byte-range GET requests for the archive but may reject HEAD
// requests from hosted CI runners. RemoteZipPointer requires HEAD only to obtain
// Content-Length, which is published in the official Zenodo metadata. Intercept
// only that metadata request; all byte-range GETs still go directly to Zenodo.
const nativeFetch = globalThis.fetch;
globalThis.fetch = async (input, init = {}) => {
  const method = String(init?.method || 'GET').toUpperCase();
  const rawUrl = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
  const url = new URL(rawUrl);
  if (method === 'HEAD' && url.hostname.endsWith('zenodo.org')) {
    return new Response(null, {
      status: 200,
      headers: { 'content-length': String(archiveBytes) },
    });
  }
  return nativeFetch(input, init);
};

console.log(`Opening remote ZIP central directory: ${archiveUrl}`);
const zip = await new RemoteZipPointer({
  url: archiveUrl,
  timeoutMs: 180000,
  requestInit: {
    cache: 'no-store',
    headers: {
      'User-Agent': 'FedSynthCT-Research-Reanalysis/1.0',
      'Accept': 'application/octet-stream,*/*',
    },
  },
}).populate();

const listing = zip.files();
const entries = listing.map((entry) => entry.filename).sort();
console.log(`Archive entries discovered: ${entries.length}`);

// Accept an optional leading directory while preserving the exact archive path.
const casePattern = /(?:^|\/)(Task1\/brain\/(1B([ABC])\d{3}))\/mr\.nii\.gz$/;
const casesByCenter = new Map();
for (const filename of entries) {
  const match = filename.match(casePattern);
  if (!match) continue;
  const directory = match[1];
  const caseId = match[2];
  const center = match[3];
  if (!casesByCenter.has(center)) casesByCenter.set(center, new Map());
  casesByCenter.get(center).set(caseId, directory);
}

if ([...casesByCenter.values()].reduce((n, m) => n + m.size, 0) === 0) {
  console.error('No expected Task1 brain paths found. First 100 archive entries:');
  console.error(entries.slice(0, 100).join('\n'));
  process.exit(2);
}

const selected = [];
for (const center of ['A', 'B', 'C']) {
  const mapping = casesByCenter.get(center) || new Map();
  const ids = [...mapping.keys()].sort();
  if (ids.length === 0) throw new Error(`No Task1 brain cases found for center ${center}`);
  selected.push({ center, caseId: ids[0], directory: mapping.get(ids[0]), availableCases: ids.length });
}
console.log('Selected deterministic technical pilot cases:', selected.map((x) => x.caseId).join(', '));

const manifest = [];
for (const item of selected) {
  const caseDir = path.join(outRoot, item.caseId);
  await fs.mkdir(caseDir, { recursive: true });
  for (const filename of ['mr.nii.gz', 'ct.nii.gz', 'mask.nii.gz']) {
    const member = `${item.directory}/${filename}`;
    if (!entries.includes(member)) throw new Error(`Missing archive member: ${member}`);
    console.log(`Fetching ${member}`);
    const bytes = await zip.fetch(member, undefined, {
      maxUncompressedSize: 1024 * 1024 * 1024,
      verifyCrc: true,
      timeoutMs: 300000,
    });
    const target = path.join(caseDir, filename);
    await fs.writeFile(target, bytes);
    manifest.push({
      center: item.center,
      case_id: item.caseId,
      member,
      target,
      bytes: bytes.byteLength,
    });
  }
}
await fs.writeFile(
  path.join(outRoot, 'extraction_manifest.json'),
  JSON.stringify({ archive: archiveUrl.toString(), archive_bytes: archiveBytes, selected, manifest }, null, 2),
);
console.log('Extraction complete.');
