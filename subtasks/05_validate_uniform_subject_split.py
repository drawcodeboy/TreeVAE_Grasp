"""Build and validate a taxonomy-balanced, subject-disjoint split.

The script searches for a global subject partition first, then samples the same
number of examples from every taxonomy label. It does not move or modify the
source ``.npz`` files; it only writes manifests, summary JSON, and SVG plots.
"""

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^subject_(?P<subject>S\d+)_.*_taxoID_(?P<label>\d+)\.npz$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "/workspace/dwkwon/HOGraspNet/processed_data/"
            "hand_pose_plus_mano_contact"
        ),
    )
    parser.add_argument("--train-per-label", type=int, default=383)
    parser.add_argument("--test-per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=50_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/taxonomy_subject_split"),
    )
    return parser.parse_args()


def collect_samples(data_path):
    samples = []
    invalid_files = []

    for path in sorted(data_path.glob("*.npz")):
        match = FILENAME_PATTERN.match(path.name)
        if match is None:
            invalid_files.append(path.name)
            continue
        samples.append(
            {
                "path": str(path.resolve()),
                "subject": match.group("subject"),
                "label": int(match.group("label")),
            }
        )

    if not samples:
        raise RuntimeError(f"No valid .npz samples found under {data_path}")
    return samples, invalid_files


def find_subject_split(samples, train_target, test_target, seed, search_trials):
    by_subject = defaultdict(Counter)
    total_by_label = Counter()
    for sample in samples:
        by_subject[sample["subject"]][sample["label"]] += 1
        total_by_label[sample["label"]] += 1

    labels = sorted(total_by_label)
    subjects = sorted(by_subject)
    required_total = train_target + test_target
    insufficient = {
        label: total_by_label[label]
        for label in labels
        if total_by_label[label] < required_total
    }
    if insufficient:
        raise RuntimeError(
            f"Labels with fewer than {required_total} total samples: {insufficient}"
        )

    rng = random.Random(seed)
    expected_test_size = round(
        len(subjects) * test_target / (train_target + test_target)
    )
    candidate_sizes = sorted(
        range(max(1, expected_test_size - 5), min(len(subjects), expected_test_size + 5) + 1),
        key=lambda size: abs(size - expected_test_size),
    )

    best = None
    trials_per_size = max(1, search_trials // len(candidate_sizes))
    for test_size in candidate_sizes:
        for _ in range(trials_per_size):
            test_subjects = frozenset(rng.sample(subjects, test_size))
            test_available = {
                label: sum(by_subject[subject][label] for subject in test_subjects)
                for label in labels
            }
            train_available = {
                label: total_by_label[label] - test_available[label]
                for label in labels
            }
            train_capacity = min(train_available.values())
            test_capacity = min(test_available.values())
            score = min(
                train_capacity / train_target,
                test_capacity / test_target,
            )

            if best is None or score > best[0]:
                best = (
                    score,
                    test_subjects,
                    train_available,
                    test_available,
                )

            if train_capacity >= train_target and test_capacity >= test_target:
                train_subjects = frozenset(set(subjects) - set(test_subjects))
                return (
                    train_subjects,
                    test_subjects,
                    train_available,
                    test_available,
                )

    _, _, best_train, best_test = best
    raise RuntimeError(
        "No feasible subject split found. "
        f"Best capacities were train={min(best_train.values())}, "
        f"test={min(best_test.values())}. Increase --search-trials or change targets."
    )


def sample_uniform_split(
    samples,
    train_subjects,
    test_subjects,
    train_target,
    test_target,
    seed,
):
    train_by_label = defaultdict(list)
    test_by_label = defaultdict(list)
    for sample in samples:
        if sample["subject"] in train_subjects:
            train_by_label[sample["label"]].append(sample)
        elif sample["subject"] in test_subjects:
            test_by_label[sample["label"]].append(sample)
        else:
            raise AssertionError(f"Unassigned subject: {sample['subject']}")

    rng = random.Random(seed)
    train_samples = []
    test_samples = []
    for label in sorted(train_by_label):
        train_samples.extend(rng.sample(train_by_label[label], train_target))
        test_samples.extend(rng.sample(test_by_label[label], test_target))

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def validate_split(train_samples, test_samples):
    train_hist = Counter(sample["label"] for sample in train_samples)
    test_hist = Counter(sample["label"] for sample in test_samples)
    train_subjects = {sample["subject"] for sample in train_samples}
    test_subjects = {sample["subject"] for sample in test_samples}
    subject_overlap = sorted(train_subjects & test_subjects)
    file_overlap = {
        sample["path"] for sample in train_samples
    } & {sample["path"] for sample in test_samples}

    train_uniform = len(set(train_hist.values())) == 1
    test_uniform = len(set(test_hist.values())) == 1
    same_labels = set(train_hist) == set(test_hist)
    passed = (
        train_uniform
        and test_uniform
        and same_labels
        and not subject_overlap
        and not file_overlap
    )
    return {
        "passed": passed,
        "train_uniform": train_uniform,
        "test_uniform": test_uniform,
        "same_labels": same_labels,
        "subject_overlap": subject_overlap,
        "file_overlap_count": len(file_overlap),
        "train_histogram": dict(sorted(train_hist.items())),
        "test_histogram": dict(sorted(test_hist.items())),
    }


def write_manifest(path, samples):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "subject", "label"])
        writer.writeheader()
        writer.writerows(samples)


def write_histogram_svg(path, histogram, title):
    width, height = 1200, 600
    left, right, top, bottom = 70, 30, 70, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    labels = sorted(histogram)
    max_count = max(histogram.values())
    slot_width = plot_width / len(labels)
    bar_width = slot_width * 0.72

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" '
        f'font-family="sans-serif" font-size="24">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="black"/>',
    ]
    for index, label in enumerate(labels):
        count = histogram[label]
        bar_height = plot_height * count / max_count
        x = left + index * slot_width + (slot_width - bar_width) / 2
        y = top + plot_height - bar_height
        elements.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="#4C78A8"/>',
                f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 20}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>',
                f'<text x="{x + bar_width / 2:.2f}" y="{max(y - 5, 55):.2f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="10">{count}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">Taxonomy label</text>',
            f'<text x="18" y="{height / 2}" text-anchor="middle" '
            f'transform="rotate(-90 18 {height / 2})" '
            f'font-family="sans-serif" font-size="16">Count</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements), encoding="utf-8")


def main():
    args = parse_args()
    samples, invalid_files = collect_samples(args.data_path)
    labels = sorted({sample["label"] for sample in samples})

    (
        train_subjects,
        test_subjects,
        train_available,
        test_available,
    ) = find_subject_split(
        samples,
        args.train_per_label,
        args.test_per_label,
        args.seed,
        args.search_trials,
    )
    train_samples, test_samples = sample_uniform_split(
        samples,
        train_subjects,
        test_subjects,
        args.train_per_label,
        args.test_per_label,
        args.seed,
    )
    validation = validate_split(train_samples, test_samples)
    if not validation["passed"]:
        raise AssertionError(f"Split validation failed: {validation}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "train_manifest.csv", train_samples)
    write_manifest(args.output_dir / "test_manifest.csv", test_samples)
    write_histogram_svg(
        args.output_dir / "train_taxonomy_histogram.svg",
        validation["train_histogram"],
        "Train taxonomy label histogram",
    )
    write_histogram_svg(
        args.output_dir / "test_taxonomy_histogram.svg",
        validation["test_histogram"],
        "Test taxonomy label histogram",
    )

    summary = {
        "seed": args.seed,
        "num_labels": len(labels),
        "labels": labels,
        "train_sample_count": len(train_samples),
        "test_sample_count": len(test_samples),
        "train_subject_count": len(train_subjects),
        "test_subject_count": len(test_subjects),
        "train_subjects": sorted(train_subjects),
        "test_subjects": sorted(test_subjects),
        "minimum_available_train_per_label": min(train_available.values()),
        "minimum_available_test_per_label": min(test_available.values()),
        "invalid_filename_count": len(invalid_files),
        "validation": validation,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Labels: {len(labels)} {labels}")
    print(f"Train: {len(train_samples)} samples, {len(train_subjects)} subjects")
    print(f"Test:  {len(test_samples)} samples, {len(test_subjects)} subjects")
    print(f"Train histogram: {validation['train_histogram']}")
    print(f"Test histogram:  {validation['test_histogram']}")
    print(f"Subject overlap: {validation['subject_overlap']}")
    print(f"Validation passed: {validation['passed']}")
    print(f"Outputs saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
