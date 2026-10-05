# Documentation

Supplementary documentation for the Qdrant memory provider. The Hermes plugin
catalog renders `README.md` at the pinned SHA, so the repo README stays the
primary document; everything here is supporting material.

## Contents

| Path | Purpose |
|------|---------|
| `DEV.md` | The technical reference — architecture, embedder backends, the markdown knowledge index, backups, the eval harness, disclosures, and what is deliberately not implemented. |
| `banner.jpg` | 2400×1200 (2:1) catalog banner, referenced by the catalog entry's `image:` field |
| `screenshots/*.png` | GitHub-hosted screenshots, pinned to the release commit |
| `competitor-analysis-entropicmem.md` | A comparison against a peer memory provider, kept for design provenance |

Architecture and hybrid-search design notes live in `DEV.md` rather than in
separate files: the unwired dense+sparse RRF path and INT8 quantization are
described there under "Not implemented", which is the only place a reader needs
to see them until they are wired in.

## Banner credit

`docs/banner.jpg` is a 2:1 centre crop of **_Parnassus_ by Anton Raphael Mengs**
(after 1761), the oil-on-panel study for the ceiling fresco of the same subject
in the Villa Albani, Rome.

- Painting: Anton Raphael Mengs (1728–1779)
- File: [File:Parnassus, by Anton Raphael Mengs.jpg](https://commons.wikimedia.org/wiki/File:Parnassus,_by_Anton_Raphael_Mengs.jpg)
- Source institution: The State Hermitage Museum, St Petersburg
- Licence: public domain (`PD-Art` / `PD-old-100-expired`) — no attribution
  required, credited here by preference

The figure is **Mnemosyne**, the Greek Titaness of memory and mother of the
nine Muses, seated to Apollo's right with her hand raised to her ear — the
attribute Ripa's *Iconologia* assigns to *mnemonics*, the art of remembering.
She is the reason this image was chosen over a generic seahorse or a logo: a
painting of the goddess of memory, by a painter who was also an engraver, in a
Neoclassical register that suits an agent's tool rather than its marketing.

The crop trims 7.2% of the source height. The card CSS
(`object-fit: cover`, `aspect-ratio: 2/1`) centre-crops any image that is not
2:1, so shipping an exact 2:1 file means the page never loses more than the
vignette we chose.

If the banner is ever regenerated: `docs/banner.jpg` is a 2400px-wide JPEG
(quality 88, stripped metadata) from a full-width 2:1 centre crop of the
Commons original (5143×2772). ImageMagick 7 was used:
`magick src.png -resize 2400x1200! -strip -quality 88 docs/banner.jpg`.

## Reference

- [Hermes memory-provider plugin guide](https://hermes-agent.nousresearch.com/docs/)
- [Qdrant documentation](https://qdrant.tech/documentation/)
