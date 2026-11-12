// Parcours complet de l'interface dans Chromium : import d'un fichier, enregistrement au micro
// (micro simulé par Chromium, qui lit fixtures/micro.wav), historique, vue système, affichage mobile
// et message d'erreur. Échoue au premier écart ; les captures d'écran vont dans screenshots/.

import assert from "node:assert/strict";
import { chromium } from "playwright";

const BASE = process.env.BASE_URL ?? "http://localhost:8080";
const OUT = "screenshots";
const FILE = "/fixtures/librispeech/1688-142285-0003.flac";
const MICRO = "/e2e/fixtures/micro.wav"; // le même énoncé, en WAV : format attendu par Chromium
const REFERENCE = "I REALLY LIKE THAT ACCOUNT OF HIMSELF BETTER THAN ANYTHING ELSE HE SAID";

/** Proportion des mots de la référence retrouvés dans l'hypothèse (approximation du 1 - WER). */
function wordRecall(reference, hypothesis) {
  const words = hypothesis.split(/\s+/);
  return reference.split(/\s+/).filter((word) => words.includes(word)).length / reference.split(/\s+/).length;
}

const browser = await chromium.launch({
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    `--use-file-for-fake-audio-capture=${MICRO}%noloop`,
  ],
});
const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ["microphone"] });
const page = await context.newPage();
const problems = [];
page.on("pageerror", (error) => problems.push(`exception : ${error.message}`));
page.on("console", (message) => {
  // Les réponses HTTP en erreur, voulues ici (422), sont signalées par Chromium : seules les autres comptent.
  if (message.type() === "error" && !message.text().startsWith("Failed to load resource")) {
    problems.push(`console : ${message.text()}`);
  }
});
const step = (name) => console.log(`- ${name}`);

step("accueil : le modèle doit être prêt");
await page.goto(BASE);
await page.getByRole("status").filter({ hasText: "Prêt" }).waitFor({ timeout: 120_000 });
await page.screenshot({ path: `${OUT}/1-accueil.png`, fullPage: true });

step("import d'un fichier FLAC puis transcription");
await page.getByRole("tab", { name: "Importer un fichier" }).click();
await page.locator('input[type="file"]').setInputFiles(FILE);
await page.getByRole("button", { name: "Transcrire" }).click();
const imported = await page.getByTestId("transcription").textContent({ timeout: 120_000 });
// Texte exact du notebook pour ce fichier (tests/regression/golden_v1.json), sur CPU comme sur GPU.
assert.equal(imported, "I REALLY LIKE THAT ACCOUNT OF HIMSELF BETTER THAN ANYTHING ELSE HE SAID");
await page.getByText("Temps de traitement").waitFor();
await page.screenshot({ path: `${OUT}/2-import-resultat.png`, fullPage: true });

step("enregistrement au micro (WebM/Opus produit par MediaRecorder)");
await page.getByRole("tab", { name: "Enregistrer" }).click();
await page.getByRole("button", { name: "Démarrer l'enregistrement" }).click();
const stop = page.getByRole("button", { name: "Arrêter" });
await stop.waitFor();
await page.waitForTimeout(2_500);
await page.screenshot({ path: `${OUT}/3-enregistrement.png`, fullPage: true });
await page.waitForTimeout(4_000);
await stop.click();
await page.getByRole("button", { name: "Transcrire" }).click();
const recorded = await page.getByTestId("transcription").textContent({ timeout: 120_000 });
const recall = wordRecall(REFERENCE, recorded);
console.log(`  transcription du micro : « ${recorded} » (${Math.round(recall * 100)} % des mots)`);
assert.ok(recall >= 0.7, `transcription du micro trop éloignée : ${recorded}`);
const format = await page.locator(".metric", { hasText: "Format reçu" }).textContent();
assert.match(format, /WebM|OGG|MP4/);
await page.screenshot({ path: `${OUT}/4-micro-resultat.png`, fullPage: true });

step("historique : les deux transcriptions, sans audio");
await page.getByRole("link", { name: "Historique" }).click();
await page.getByRole("heading", { name: "Historique" }).waitFor();
assert.equal(await page.locator("li.entry").count(), 2);
await page.screenshot({ path: `${OUT}/5-historique.png`, fullPage: true });

step("vue système : périphérique, modèle, limites");
await page.getByRole("link", { name: "Système" }).click();
await page.getByRole("heading", { name: "Périphérique de calcul" }).waitFor();
await page.getByText("Versions présentes").waitFor();
await page.screenshot({ path: `${OUT}/6-systeme.png`, fullPage: true });

step("fichier illisible : message d'erreur explicite");
await page.getByRole("link", { name: "Transcription" }).click();
await page.getByRole("tab", { name: "Importer un fichier" }).click();
await page.locator('input[type="file"]').setInputFiles({ name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("pas un son") });
await page.getByRole("button", { name: "Transcrire" }).click();
const alert = page.getByRole("alert").filter({ hasText: "illisible" });
await alert.waitFor({ timeout: 30_000 });
assert.match(await alert.textContent(), /INVALID_AUDIO/);
await page.screenshot({ path: `${OUT}/7-erreur.png`, fullPage: true });

step("affichage mobile");
const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true });
const phone = await mobile.newPage();
await phone.goto(BASE);
await phone.getByRole("status").filter({ hasText: "Prêt" }).waitFor({ timeout: 60_000 });
await phone.screenshot({ path: `${OUT}/8-mobile.png`, fullPage: true });
const overflow = await phone.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
assert.ok(overflow <= 0, `défilement horizontal sur mobile : ${overflow} px`);

await browser.close();
assert.deepEqual(problems, [], `erreurs dans le navigateur :\n${problems.join("\n")}`);
console.log("Parcours réussi.");
