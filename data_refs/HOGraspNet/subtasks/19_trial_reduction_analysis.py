import sys, os
sys.path.append(os.getcwd())

import glob
from collections import defaultdict

# DenseHOGraspNet 포인트클라우드 데이터 경로
DENSEHOG_DATA_PATH = "data/densehograspnet_pointcloud"

def analyze_trial_reduction():
    """Trial별 데이터를 분석하고, 각 조합에서 하나씩만 선택했을 때 개수 변화를 봅니다."""
    
    npz_files = glob.glob(os.path.join(DENSEHOG_DATA_PATH, "**", "*.npz"), recursive=True)
    
    print(f"Total npz files: {len(npz_files)}")
    print("=" * 80)
    
    # 파일명 패턴 분석: trial_X__subject__subject_idx__grasp_GraspType_pc.npz
    # 예: trial_0__sub3__sub3_56__14_Tripod_pc.npz
    
    unique_combinations = set()
    trial_groups = defaultdict(list)
    subject_session_grasp_object = defaultdict(list)
    
    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        # trial_0__sub3__sub3_56__14_Tripod_pc.npz
        try:
            parts = filename.replace('_pc.npz', '').split('__')
            if len(parts) >= 3:
                trial_part = parts[0]  # trial_0
                subject_part = parts[1]  # sub3
                subject_idx_part = parts[2]  # sub3_56
                grasp_part = parts[3] if len(parts) > 3 else ""  # 14_Tripod
                
                trial_num = trial_part.split('_')[1]
                grasp_num = grasp_part.split('_')[0]
                grasp_name = '_'.join(grasp_part.split('_')[1:])
                
                # (subject, subject_idx, grasp_num, grasp_name) 조합 만들기
                key = (subject_part, subject_idx_part, grasp_num, grasp_name)
                unique_combinations.add(key)
                trial_groups[key].append((trial_num, npz_path))
                
        except Exception as e:
            pass
    
    print(f"Unique (subject, idx, grasp) combinations: {len(unique_combinations)}")
    print("\nTrial distribution per combination:")
    
    trial_counts = defaultdict(int)
    for key, files in trial_groups.items():
        trial_counts[len(files)] += 1
    
    print(f"\nCombinations with N trials:")
    for num_trials in sorted(trial_counts.keys()):
        count = trial_counts[num_trials]
        total_for_this = count * num_trials
        total_if_one = count * 1
        print(f"  {num_trials} trial(s): {count:6d} combinations ({total_for_this:7d} files -> {total_if_one:7d} if picking 1 each)")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Original total files: {len(npz_files):,}")
    print(f"Unique (subject, idx, grasp) combinations: {len(unique_combinations):,}")
    print(f"\nIf you pick 1 trial per combination: ~{len(unique_combinations):,} files")
    print(f"Reduction ratio: {len(npz_files) / len(unique_combinations):.1f}x (keeping {100*len(unique_combinations)/len(npz_files):.1f}%)")

if __name__ == '__main__':
    analyze_trial_reduction()
