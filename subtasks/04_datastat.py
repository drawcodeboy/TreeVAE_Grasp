import os, sys

import matplotlib.pyplot as plt

def main():
    data_path = "/workspace/dwkwon/HOGraspNet/processed_data/hand_pose_plus_mano_contact"
    taxo_hist = [0 for i in range(0, 33+1)]
    for filename in os.listdir(data_path):
        taxo_label = int(filename.split('.')[0].split('_')[-1])
        taxo_hist[taxo_label] += 1

    nonzero_hist = [
        (label, count)
        for label, count in enumerate(taxo_hist)
        if count != 0
    ]
    if nonzero_hist:
        min_count = min(count for _, count in nonzero_hist)
        max_count = max(count for _, count in nonzero_hist)
        min_labels = [label for label, count in nonzero_hist if count == min_count]
        max_labels = [label for label, count in nonzero_hist if count == max_count]

        print(f"Non-zero min count: {min_count} (labels: {min_labels})")
        print(f"Non-zero max count: {max_count} (labels: {max_labels})")
    else:
        print("No non-zero taxonomy label counts found.")

    labels = list(range(len(taxo_hist)))
    plt.figure(figsize=(12, 6))
    plt.bar(labels, taxo_hist, width=0.8)
    plt.title("Taxonomy label histogram")
    plt.xlabel("Taxonomy label")
    plt.ylabel("Count")
    plt.xticks(labels)
    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)
    output_path = os.path.join("assets", "taxonomy_label_histogram.png")
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Histogram saved to: {output_path}")

    # Non-zero min count: 483 (labels: [23])
    # Non-zero max count: 5634 (labels: [12])


if __name__ == '__main__':
    main()