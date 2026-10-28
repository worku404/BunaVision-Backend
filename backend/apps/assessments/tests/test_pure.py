"""Framework-free tests: run with `python -m unittest apps.assessments.tests.test_pure`."""

import io
import unittest

from PIL import Image

from apps.assessments.analysis import Detection, limitations_for, summarize
from apps.assessments.clients import rule_based_advisory
from apps.assessments.image_validation import ImageFormat, ImageLimits, validate_image
from apps.assessments.taxonomy import TAXONOMY, DefectCategory, DefectClass, DefectTaxonomy, UnknownDefectLabel
from apps.common.errors import (
    ERROR_CODES, AppError, ClientError, DependencyError, ImageTooLarge, InvalidImage, NotCoffeeImage,
    UnsupportedImageFormat,
)

LIMITS = ImageLimits(max_bytes=10 * 1024 * 1024, max_pixels=1_000_000, min_side_px=320)


def make_image(fmt: str = "JPEG", size: tuple[int, int] = (400, 400)) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise(size, 64).convert("RGB").save(buffer, format=fmt)
    return buffer.getvalue()


def detection(label: str, confidence: float) -> Detection:
    return Detection(TAXONOMY.from_vision_label(label), confidence, (0.0, 0.0, 1.0, 1.0))


class TaxonomyTests(unittest.TestCase):
    def test_inventory_and_coverage(self) -> None:
        categories = [c.category for c in TAXONOMY.all]
        self.assertEqual(
            (len(categories), categories.count(DefectCategory.PRIMARY), categories.count(DefectCategory.SECONDARY)),
            (13, 6, 7),
        )
        self.assertEqual({c.vision_label for c in TAXONOMY.scanned}, {"Full Black", "Partial Black", "Coffee Husk / Parchment"})
        self.assertEqual(len(TAXONOMY.not_scanned), 10)

    def test_unknown_label_is_an_error_not_a_guess(self) -> None:
        with self.assertRaises(UnknownDefectLabel):
            TAXONOMY.from_vision_label("Mystery Spot")

    def test_duplicates_are_rejected(self) -> None:
        a = DefectClass("a", "A", DefectCategory.PRIMARY, vision_label="x")
        with self.assertRaises(ValueError):
            DefectTaxonomy([a, DefectClass("a", "B", DefectCategory.PRIMARY)])


class AnalysisTests(unittest.TestCase):
    def test_zero_detections_is_a_valid_summary(self) -> None:
        summary = summarize([])
        self.assertEqual((summary["detection_count"], summary["by_class"]), (0, []))

    def test_counts_are_detections_grouped_by_class(self) -> None:
        summary = summarize([detection("Full Black", 0.9), detection("Partial Black", 0.9), detection("Partial Black", 0.5)])
        self.assertEqual(
            (summary["detection_count"], summary["primary_detection_count"], summary["secondary_detection_count"]),
            (3, 1, 2),
        )
        top = summary["by_class"][0]
        self.assertEqual((top["defect"]["name"], top["count"], top["mean_confidence"]), ("Partial Black", 2, 0.7))

    def test_coverage_numbers_come_from_the_taxonomy(self) -> None:
        message = {x["code"]: x["message"] for x in limitations_for(TAXONOMY.scanned, TAXONOMY.not_scanned)}[
            "PARTIAL_CLASS_COVERAGE"
        ]
        self.assertIn("Only 3 of 13", message)
        self.assertIn("other 10", message)

    def test_no_coverage_warning_when_everything_is_scanned(self) -> None:
        codes = {x["code"] for x in limitations_for(TAXONOMY.all, ())}
        self.assertNotIn("PARTIAL_CLASS_COVERAGE", codes)


class ImageValidationTests(unittest.TestCase):
    def test_accepts_supported_formats(self) -> None:
        for fmt, expected in (("JPEG", ImageFormat.JPEG), ("PNG", ImageFormat.PNG), ("WEBP", ImageFormat.WEBP)):
            self.assertIs(validate_image(make_image(fmt), LIMITS).format, expected)

    def test_rejects_garbage_gif_and_truncated(self) -> None:
        for data, error in (
            (b"", InvalidImage), (b"not an image", InvalidImage), (make_image("GIF"), UnsupportedImageFormat),
            (make_image("JPEG")[:2000], InvalidImage),
        ):
            with self.assertRaises(error):
                validate_image(data, LIMITS)

    def test_limits(self) -> None:
        with self.assertRaises(ImageTooLarge):
            validate_image(make_image(), ImageLimits(100, 10**9, 1))
        with self.assertRaises(InvalidImage):
            validate_image(make_image(size=(400, 400)), ImageLimits(10**7, 100_000, 10))
        with self.assertRaises(InvalidImage):
            validate_image(make_image(size=(100, 100)), LIMITS)


class ErrorTests(unittest.TestCase):
    def test_every_error_class_uses_a_listed_code(self) -> None:
        def all_subclasses(cls: type[AppError]) -> list[type[AppError]]:
            return [cls] + [s for sub in cls.__subclasses__() for s in all_subclasses(sub)]

        for cls in all_subclasses(AppError):
            self.assertIn(cls.code, ERROR_CODES, cls.__name__)
            self.assertTrue(400 <= cls.http_status <= 599, cls.__name__)

    def test_client_and_dependency_families_are_disjoint(self) -> None:
        self.assertFalse(issubclass(NotCoffeeImage, DependencyError))
        self.assertTrue(issubclass(NotCoffeeImage, ClientError))

    def test_not_coffee_carries_is_coffee_false(self) -> None:
        error = NotCoffeeImage("gate", coffee_confidence=0.02, threshold=0.75)
        self.assertEqual(error.details, {"is_coffee": False, "coffee_confidence": 0.02, "threshold": 0.75})


class RuleBasedAdvisoryTests(unittest.TestCase):
    def test_zero_defects_returns_reassurance_and_verification_advice(self) -> None:
        summary = {"detection_count": 0, "primary_detection_count": 0, "secondary_detection_count": 0}
        advisory = rule_based_advisory.generate(summary, TAXONOMY.scanned, TAXONOMY.not_scanned)
        self.assertEqual(advisory["generator_model"], "rule-based-v1")
        self.assertIn("No visible defect detections", advisory["summary"]["en"])
        self.assertIn("ምንም የሚታይ ጉድለት አልተገኘም", advisory["summary"]["am"])
        self.assertEqual(len(advisory["recommendations"]), 1)
        self.assertIn("manual verification", advisory["recommendations"][0]["en"].lower())

    def test_primary_defects_returns_high_severity_advice(self) -> None:
        summary = {"detection_count": 3, "primary_detection_count": 1, "secondary_detection_count": 2}
        advisory = rule_based_advisory.generate(summary, TAXONOMY.scanned, TAXONOMY.not_scanned)
        self.assertEqual(advisory["generator_model"], "rule-based-v1")
        self.assertIn("including 1 primary defect(s)", advisory["summary"]["en"])
        self.assertIn("1 ዋና ጉድለቶች", advisory["summary"]["am"])
        self.assertEqual(len(advisory["recommendations"]), 2)
        self.assertIn("hand-pick", advisory["recommendations"][0]["en"].lower())

    def test_secondary_defects_only_returns_sorting_advice(self) -> None:
        summary = {"detection_count": 4, "primary_detection_count": 0, "secondary_detection_count": 4}
        advisory = rule_based_advisory.generate(summary, TAXONOMY.scanned, TAXONOMY.not_scanned)
        self.assertEqual(advisory["generator_model"], "rule-based-v1")
        self.assertIn("4 secondary defect(s)", advisory["summary"]["en"])
        self.assertIn("4 ሁለተኛ ደረጃ ጉድለቶች", advisory["summary"]["am"])
        self.assertEqual(len(advisory["recommendations"]), 2)
        self.assertIn("huller screens", advisory["recommendations"][0]["en"].lower())

    def test_advisory_structure_matches_bilingual_contract(self) -> None:
        summary = {"detection_count": 1, "primary_detection_count": 1, "secondary_detection_count": 0}
        advisory = rule_based_advisory.generate(summary, TAXONOMY.scanned, TAXONOMY.not_scanned)
        self.assertIn("en", advisory["summary"])
        self.assertIn("am", advisory["summary"])
        for rec in advisory["recommendations"]:
            self.assertIn("en", rec)
            self.assertIn("am", rec)
            self.assertTrue(len(rec["en"]) > 0)
            self.assertTrue(len(rec["am"]) > 0)


if __name__ == "__main__":
    unittest.main()

