# Sample Test Images

This folder is intended to hold representative skin wound / disease photos for
local testing and hackathon demonstration.

## How to add images

1. Photograph or download **royalty-free** clinical images of the conditions below.
2. Save them as `PNG` or `JPG`, named descriptively (e.g. `contact_dermatitis_arm.jpg`).
3. Reference them in the Gradio `Examples` section in `app.py` by updating the
   `examples` list with the file path.

> **Important — Privacy & Licensing**
> - Only use images you own, have explicit permission to use, or that are
>   released under a CC0 / public-domain licence.
> - Do **not** upload real patient photos without proper consent and de-identification.
> - Suggested free sources: [Wikimedia Commons](https://commons.wikimedia.org) (medical category),
>   [DermNet NZ](https://dermnetnz.org) (check licence per image).

---

## Conditions the model analyzes

| # | Condition | Typical Severity | Visual Cues |
|---|---|---|---|
| 1 | **Superficial Abrasion / Cut** | Low | Shallow skin break, mild redness, minimal bleeding |
| 2 | **Contact Dermatitis** | Low – Medium | Red, itchy patch; may show blistering or oozing |
| 3 | **Atopic Eczema (Flare)** | Medium | Dry, thickened skin; lichenification; excoriation marks |
| 4 | **Tinea Corporis (Ringworm)** | Low | Ring-shaped, scaly, red border; central clearing |
| 5 | **Partial-Thickness Burn (2nd degree)** | High | Blistering, moist, red/pink base; extremely painful |
| 6 | **Psoriasis Plaque** | Medium | Well-defined silvery-white scales on red base |
| 7 | **Cellulitis / Skin Infection** | Urgent | Spreading redness, warmth, swelling, possible fever |

---

## Naming convention

```
<condition_slug>_<body_part>_<index>.<ext>
# Examples:
contact_dermatitis_arm_01.jpg
burn_2nd_degree_hand_01.jpg
cellulitis_leg_01.jpg
```

---

*MediVision — AMD Developer Hackathon 2026*
