// Rend kirari.svg en kirari.png (256 px, Linux) et kirari.ico (16 à 256 px, Windows, images PNG dedans).
// Demande Playwright avec son Chromium : node tools/release/logo/render.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const { chromium } = createRequire(import.meta.url)("playwright");
const ici = dirname(fileURLToPath(import.meta.url));
const release = dirname(ici);
const tailles = [16, 24, 32, 48, 64, 128, 256];
const svg = readFileSync(join(ici, "kirari.svg"), "utf8");

const navigateur = await chromium.launch();
const page = await navigateur.newPage();
const images = [];
for (const t of tailles) {
  await page.setViewportSize({ width: t, height: t });
  await page.setContent(
    "<style>html,body{margin:0;background:transparent}svg{display:block;width:100vw;height:100vh}</style>" + svg);
  images.push(await page.screenshot({ omitBackground: true }));
}
await navigateur.close();

writeFileSync(join(release, "linux", "kirari.png"), images.at(-1));

// ICO : en-tête de 6 octets, une entrée de 16 octets par taille, puis les PNG à la suite.
const tete = Buffer.alloc(6 + 16 * tailles.length);
tete.writeUInt16LE(0, 0);
tete.writeUInt16LE(1, 2);
tete.writeUInt16LE(tailles.length, 4);
let decalage = tete.length;
tailles.forEach((t, i) => {
  const e = 6 + 16 * i;
  tete.writeUInt8(t >= 256 ? 0 : t, e);
  tete.writeUInt8(t >= 256 ? 0 : t, e + 1);
  tete.writeUInt16LE(1, e + 4);  // plans
  tete.writeUInt16LE(32, e + 6);  // bits par pixel
  tete.writeUInt32LE(images[i].length, e + 8);
  tete.writeUInt32LE(decalage, e + 12);
  decalage += images[i].length;
});
writeFileSync(join(release, "windows", "kirari.ico"), Buffer.concat([tete, ...images]));
