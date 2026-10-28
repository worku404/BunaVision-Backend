"""Rule-based advisory generator used when Gemini is unreachable."""

from collections.abc import Sequence
from typing import Any

from apps.assessments.taxonomy import DefectClass

GENERATOR_MODEL = "rule-based-v1"


def generate(
    summary: dict[str, Any],
    scanned: Sequence[DefectClass],
    not_scanned: Sequence[DefectClass],
) -> dict[str, Any]:
    total_defects = summary.get("detection_count", 0)
    primary_defects = summary.get("primary_detection_count", 0)
    secondary_defects = summary.get("secondary_detection_count", 0)

    # 1. No defects detected in scanned classes
    if total_defects == 0:
        return {
            "generator_model": GENERATOR_MODEL,
            "summary": {
                "en": (
                    "No visible defect detections were found for the scanned classes. "
                    "However, this preliminary screening does not guarantee the sample is free from defects."
                ),
                "am": (
                    "ከተመረመሩት የጉድለት አይነቶች ውስጥ ምንም የሚታይ ጉድለት አልተገኘም። "
                    "ይሁን እንጂ ይህ የመጀመሪያ ደረጃ ምርመራ ናሙናው ሙሉ በሙሉ ከጉድለት ነፃ መሆኑን አያረጋግጥም።"
                ),
            },
            "recommendations": [
                {
                    "en": "Perform manual verification with a certified coffee grader before export preparation.",
                    "am": "ወደ ውጭ ከመላክዎ በፊት ከተረጋገጠ የቡና ጥራት መርማሪ ጋር በእጅ ምርመራ ያድርጉ።",
                }
            ],
        }

    # 2. Primary defects detected (e.g. Full Black) - High severity
    if primary_defects > 0:
        return {
            "generator_model": GENERATOR_MODEL,
            "summary": {
                "en": (
                    f"A total of {total_defects} defect detections were identified, including {primary_defects} "
                    "primary defect(s). Primary defects significantly impact final grade and cup quality."
                ),
                "am": (
                    f"በአጠቃላይ {total_defects} የጉድለት ምልክቶች የተገኙ ሲሆን፣ {primary_defects} ዋና ጉድለቶች ይገኙበታል። "
                    "ዋና ጉድለቶች የቡናውን ጥራት ደረጃ እና ጣዕም በእጅጉ ይቀንሳሉ።"
                ),
            },
            "recommendations": [
                {
                    "en": "Immediately hand-pick and remove black and heavily discolored beans from the lot.",
                    "am": "ጥቁር እና የተበላሹ ፍሬዎችን ወዲያውኑ ከቡናው ውስጥ በእጅ ለይተው ያስወግዱ።",
                },
                {
                    "en": "Review fermentation and drying bed moisture levels to prevent further bean blackening.",
                    "am": "ተጨማሪ ጥቁረት እንዳይከሰት የማብላያ እና የማድረቂያ አልጋዎች የእርጥበት መጠንን ይፈትሹ።",
                },
            ],
        }

    # 3. Secondary defects only (e.g. Partial Black, Husk / Parchment)
    return {
        "generator_model": GENERATOR_MODEL,
        "summary": {
            "en": (
                f"Detected {total_defects} secondary defect(s) in the sample. "
                "Secondary defects can reduce the overall grade if not addressed."
            ),
            "am": (
                f"በናሙናው ውስጥ {total_defects} ሁለተኛ ደረጃ ጉድለቶች ተገኝተዋል። "
                "እነዚህ ጉድለቶች ካልተወገዱ የቡናውን ደረጃ ዝቅ ሊያደርጉ ይችላሉ።"
            ),
        },
        "recommendations": [
            {
                "en": "Adjust huller screens and blower settings to remove parchment and husk fragments.",
                "am": "የገፈፋ እና የመንፊያ ማሽኖችን በማስተካከል የቆዳ እና የገፈፋ ቅሪቶችን ያስወግዱ።",
            },
            {
                "en": "Sort the batch through gravity tables or screen sieves to eliminate partially defective beans.",
                "am": "በከፊል የተበላሹ ፍሬዎችን ለማስወገድ በስበት ማበጠሪያ ወይም በወንፊት ያጣሩ።",
            },
        ],
    }
