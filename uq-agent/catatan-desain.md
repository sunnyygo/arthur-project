# Catatan Keputusan Desain — UQ Agent
(dipindahkan dari percakapan; bahan Bab III & sidang)

1. **Alpha = 0.4** (bobot verbalized confidence). Konsistensi diberi bobot lebih
   besar (0.6) karena Tian dkk. (2023) membuktikan verbalized confidence
   sistematis overconfident — pengakuan lebih lemah daripada bukti perilaku.
2. **Ambang alert: 0.75 / 0.50.** Dibekukan sebelum eksperimen (prinsip P4 rubrik
   ground truth). ≥0.75 OK; 0.50–0.74 PERLU CEK MANUAL; <0.50 TIDAK PASTI.
3. **Temperature 0.7.** Wajib >0 agar pengulangan tidak identik (self-consistency
   tidak bermakna pada model deterministik). Nilai tercatat di setiap trace JSON.
4. **3 pengulangan** (bukan 5):Proposal menetapkan 3× — biaya 1 langkah = 4
   pemanggilan pipeline + 1 verbalized = 5 total.
5. **VC gagal parsing → 0.5 (netral).** Aturan konservatif yang tidak
   menguntungkan sistem yang diuji (sejalan aturan rubrik: beban pembuktian
   pada agent).
6. **Pembanding konsistensi per tahap** (sesuai penjelasan-alur-sistem-uq.md):
   pencarian = Jaccard irisan sumber; ekstraksi = kesamaan persis klaim;
   sintesis = kemiripan makna (Jaccard token — versi ringan tanpa
   sentence-transformers, cukup untuk VPS 2GB); penyimpulan = arah kesimpulan.
7. **Mock backend meniru perilaku LLM asli**: deterministik pada temperature 0,
   bervariasi pada temperature >0 — agar unit test merepresentasikan kasus nyata.
8. **UI ringan tanpa framework JS** (vanilla) — sesuai batasan "tidak membangun
   frontend penuh" di tumpukan-teknologi-dan-coding.md; dashboard confidence
   trace cukup HTML+fetch.

## v1.1 — Grounding ekstraksi (2026-09-06, hasil pilot run)

**Perubahan:** prompt tahap EKSTRAKSI kini mewajibkan setiap klaim disertai field
`quote` berisi kutipan verbatim (min. 8 kata berturut-turut) dari abstrak sumber.
Hasil pencarian API kini menyertakan abstrak asli (arXiv `summary`, Crossref
`abstract`, OpenAlex `abstract_inverted_index` direkonstruksi, dipotong 600 char).

**Alasan (temuan pilot):** pada run pilot "AI agent trend detection", 4 pengulangan
ekstraksi menghasilkan klaim bermakna identik namun berbeda rumusan kata
("example-based prompt learning" vs "pembelajaran prompt berbasis contoh") —
metrik konsistensi persis-string menilainya 0.25 padahal maknanya 4/4 sama.
Grounding memaksa klaim menempel ke teks sumber asli, menekan parafrase liar
sekaligus memberi bukti kutipan yang bisa diverifikasi manusia.

**Yang TIDAK berubah (alat ukur tetap dibekukan):** ALPHA=0.4, threshold 0.75/0.50,
N_REPEAT=3, temperature=0.7, metrik per tahap, prinsip penyadap pasif.
Status: DIBEKUKAN ULANG untuk eksperimen formal sejak commit ini. Sebelum
eksperimen formal, tidak ada lagi perubahan engine/prompt/metrik tanpa
pencatatan versi baru di file ini.
