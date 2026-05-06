"""
Korean (Seoul Corpus / SLR113) → IPA → Lee-Hon 39 phoneme mapping.

Mirrors dutch_preprocess.py: a manual source-language → IPA dict, then panphon
articulatory feature distance to find the closest LH39 phoneme. No edits to
utils.py or any model file are needed; predict.py / test_results.py just see
LH39-mappable phoneme labels in the .phn / .wrd files after preprocessing.

Public API mirrors dutch_preprocess:
    aligner_pipeline(label) -> [{"ipa_part": str, "lh39": str, "dist": float}, ...]
"""
import re
import panphon
import panphon.distance

from dutch_preprocess import LH39_IPA, timit_leehon_39_phonemes, find_best_leehon39

ft = panphon.FeatureTable()
dst = panphon.distance.Distance()

# Seoul Corpus phoneme codes (2 chars each, except 'sil') → IPA.
# Korean 3-way obstruent contrast: lax / aspirated / tense.
KOREAN_TO_IPA = {
    # Plosives
    "k0": "k",   "kh": "kʰ",  "kk": "k͈",
    "p0": "p",   "ph": "pʰ",  "pp": "p͈",
    "t0": "t",   "th": "tʰ",  "tt": "t͈",
    # Affricates (alveolo-palatal)
    "c0": "tɕ",  "ch": "tɕʰ", "cc": "t͈ɕ",
    # Fricatives
    "s0": "s",   "ss": "s͈",
    "hh": "h",
    # Nasals
    "mm": "m",   "nn": "n",   "ng": "ŋ",
    # Liquid
    "ll": "l",
    # Monophthong vowels
    "aa": "a",   "ee": "e",   "ii": "i",
    "oo": "o",   "uu": "u",   "vv": "ʌ",   "xx": "ɯ",
    # Palatalized (y-) vowels
    "ya": "ja",  "ye": "je",  "yo": "jo",  "yu": "ju",  "yv": "jʌ",
    # Labialized (w-) vowels
    "wa": "wa",  "we": "we",  "wi": "wi",  "wv": "wʌ",
    # Diphthong
    "xi": "ɯi",
    # Silence
    "sil": "sil",
}


def _split_compound(label: str):
    """Split a Seoul-Corpus label into 2-char phoneme codes.

    Handles single phonemes ('vv'), syllable-hyphenated forms ('mmuu-s0xxnn'),
    and the literal 'sil' token. Returns a list of phoneme codes.
    """
    if label.lower() == "sil":
        return ["sil"]
    cleaned = label.replace("-", "")
    if not cleaned:
        return []
    # Korean phonemes are 2 chars each (except 'sil', already handled).
    return [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)]


def get_ipa_from_korean(korean_label: str):
    """Korean Seoul-Corpus label → list of IPA segments.

    Korean codes are checked BEFORE the LH39 short-circuit because a few 2-char
    Korean codes collide with English LH39 names but represent different sounds
    (notably 'th' = Korean /tʰ/ vs English /θ/, 'ch' = Korean /tɕʰ/ vs English /tʃ/).
    """
    label = korean_label.lower()
    if label in ("sil", "h#"):
        return ["sil"]
    # Korean single-phoneme codes (must precede the LH39 passthrough)
    if label in KOREAN_TO_IPA:
        return [KOREAN_TO_IPA[label]]
    # Already an LH39 phoneme (e.g., dataset mixes English labels): use as-is
    if label in timit_leehon_39_phonemes:
        return [label]
    # Compound label like 'mmuu-s0xxnn' — strip hyphens, split into 2-char codes
    parts = _split_compound(korean_label)
    return [KOREAN_TO_IPA.get(p, p) for p in parts]


def aligner_pipeline(korean_input: str):
    """Korean label → list of {ipa_part, lh39, dist} dicts (one per phoneme)."""
    ipa_segments = get_ipa_from_korean(korean_input)
    results = []
    for ipa_seg in ipa_segments:
        match, d = find_best_leehon39(ipa_seg)
        results.append({"ipa_part": ipa_seg, "lh39": match, "dist": d})
    return results


if __name__ == "__main__":
    # Smoke-test on a handful of representative inputs.
    cases = ["vv", "mmuu-s0xxnn", "vvmm-mmaa-k0aa", "sil", "ya", "k0", "s0"]
    for case in cases:
        print(f"\nINPUT: {case}")
        for item in aligner_pipeline(case):
            print(f"  '{item['ipa_part']}' -> {item['lh39']} (dist {item['dist']})")
