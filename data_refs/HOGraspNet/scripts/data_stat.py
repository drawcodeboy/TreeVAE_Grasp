import os, sys
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

def main():
    base_path = "/workspace/lab_intern/KDW/HOGraspNet/data/labeling_data"

    subject_counts = defaultdict(int)
    grasp_counts = defaultdict(int)
    obj_counts = defaultdict(int)

    for item in os.listdir(base_path):
        subject_name = item.split("_")[1][1:]
        obj_name = item.split("_")[3]
        grasp_name = item.split("_")[-1]
        subject_counts[int(subject_name)] += 1
        obj_counts[int(obj_name)] += 1
        grasp_counts[int(grasp_name)] += 1
        

    subject_counts = dict(sorted(subject_counts.items()))
    obj_counts = dict(sorted(obj_counts.items()))
    grasp_counts = dict(sorted(grasp_counts.items()))

    plt.figure(figsize=(24, 12))
    plt.subplot(3, 1, 1)
    num_subjects = 0
    for key, count in subject_counts.items():
        print(f"Subject {key}: {count} files")
        num_subjects += 1
    plt.bar(subject_counts.keys(), subject_counts.values())
    plt.xticks(list(subject_counts.keys()))
    plt.xlabel("Subject ID")
    plt.ylabel("Number of Files")
    plt.title("Number of Files per Subject")
    print(f"Total number of subjects: {num_subjects}")

    plt.subplot(3, 1, 2)
    num_objects = 0
    for key, count in obj_counts.items():
        print(f"Object {key}: {count} files")
        num_objects += 1
    plt.bar(obj_counts.keys(), obj_counts.values())
    plt.xticks(list(obj_counts.keys()))
    plt.xlabel("Object ID")
    plt.ylabel("Number of Files")
    plt.title("Number of Files per Object")
    print(f"Total number of objects: {num_objects}")

    plt.subplot(3, 1, 3)
    num_grasps = 0
    for key, count in grasp_counts.items():
        print(f"Grasp {key}: {count} files")
        num_grasps += 1
    plt.bar(grasp_counts.keys(), grasp_counts.values())
    plt.xticks(list(grasp_counts.keys()))
    plt.xlabel("Grasp ID")
    plt.ylabel("Number of Files")
    plt.title("Number of Files per Grasp")
    print(f"Total number of grasps: {num_grasps}")

    plt.tight_layout()
    os.makedirs("vis/", exist_ok=True)
    plt.savefig("vis/data_stat.png")

    # basepath의 용량
    # total_size = 0
    # for root, dirs, files in os.walk(base_path):
    #     for filename in files:
    #         filepath = os.path.join(root, filename)
    #         total_size += os.path.getsize(filepath)
    # print(f"Total size of {base_path}: {total_size / (1024 * 1024 * 1024):.2f} GB")

if __name__ == "__main__":
    main()