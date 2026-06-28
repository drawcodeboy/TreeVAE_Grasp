import sys, os
sys.path.append(os.getcwd())

import numpy as np
import glob
from pathlib import Path

# DenseHOGraspNet 데이터 경로
DENSEHOG_DATA_PATH = "data/densehograspnet_pointcloud"

def analyze_densehograspnet_scales():
    """DenseHOGraspNet 포인트클라우드의 스케일 및 좌표 범위를 분석합니다."""
    
    # 모든 npz 파일 찾기
    npz_files = glob.glob(os.path.join(DENSEHOG_DATA_PATH, "**", "*.npz"), recursive=True)
    
    print(f"Found {len(npz_files)} npz files")
    print("=" * 80)
    
    all_object_ranges = []
    all_hand_ranges = []
    object_scales = []
    hand_scales = []
    
    for idx, npz_path in enumerate(npz_files, 1):
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # 필요한 키 확인
            if 'object_pc' not in data or 'hand_pc' not in data:
                print(f"[{idx:05d}] SKIP: Missing keys - {os.path.basename(npz_path)}")
                continue
            
            object_pc = data['object_pc']
            hand_pc = data['hand_pc']
            
            # 포인트클라우드 정규화 (단일 샘플인 경우 처리)
            if object_pc.ndim == 3:
                object_pc = object_pc[0]
            if hand_pc.ndim == 3:
                hand_pc = hand_pc[0]
            
            # 좌표 범위 분석
            obj_min, obj_max = object_pc.min(axis=0), object_pc.max(axis=0)
            hand_min, hand_max = hand_pc.min(axis=0), hand_pc.max(axis=0)
            
            obj_range = obj_max - obj_min
            hand_range = hand_max - hand_min
            
            # 스케일 (최대 범위)
            obj_scale = obj_range.max()
            hand_scale = hand_range.max()
            
            all_object_ranges.append(obj_range)
            all_hand_ranges.append(hand_range)
            object_scales.append(obj_scale)
            hand_scales.append(hand_scale)
            
            # 이상 케이스 체크
            if obj_scale > 2.0 or hand_scale > 2.0:
                print(f"[{idx:05d}] OUTLIER: Object_scale={obj_scale:.4f}, Hand_scale={hand_scale:.4f}")
                print(f"           File: {os.path.basename(npz_path)}")
                print(f"           Object range: {obj_range}, Hand range: {hand_range}")
            
            if idx % 100 == 0:
                print(f"[{idx:05d}] Processed - Object_scale={obj_scale:.4f}, Hand_scale={hand_scale:.4f}")
                
        except Exception as e:
            print(f"[{idx:05d}] ERROR: {e} - {os.path.basename(npz_path)}")
            continue
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if len(object_scales) > 0:
        object_scales = np.array(object_scales)
        hand_scales = np.array(hand_scales)
        
        print(f"\nTotal processed samples: {len(object_scales)}")
        
        print(f"\nObject PC Scales (max dimension range):")
        print(f"  Min: {object_scales.min():.6f}")
        print(f"  Max: {object_scales.max():.6f}")
        print(f"  Mean: {object_scales.mean():.6f}")
        print(f"  Std: {object_scales.std():.6f}")
        print(f"  Median: {np.median(object_scales):.6f}")
        print(f"  Q1 (25%): {np.percentile(object_scales, 25):.6f}")
        print(f"  Q3 (75%): {np.percentile(object_scales, 75):.6f}")
        print(f"  Samples > 1.0: {(object_scales > 1.0).sum()}")
        print(f"  Samples > 2.0: {(object_scales > 2.0).sum()}")
        
        print(f"\nHand PC Scales (max dimension range):")
        print(f"  Min: {hand_scales.min():.6f}")
        print(f"  Max: {hand_scales.max():.6f}")
        print(f"  Mean: {hand_scales.mean():.6f}")
        print(f"  Std: {hand_scales.std():.6f}")
        print(f"  Median: {np.median(hand_scales):.6f}")
        print(f"  Q1 (25%): {np.percentile(hand_scales, 25):.6f}")
        print(f"  Q3 (75%): {np.percentile(hand_scales, 75):.6f}")
        print(f"  Samples > 1.0: {(hand_scales > 1.0).sum()}")
        print(f"  Samples > 2.0: {(hand_scales > 2.0).sum()}")
        
        print(f"\nOverall Scale Range (both object and hand):")
        all_scales = np.concatenate([object_scales, hand_scales])
        print(f"  Min: {all_scales.min():.6f}")
        print(f"  Max: {all_scales.max():.6f}")
        print(f"  Mean: {all_scales.mean():.6f}")
        print(f"  Std: {all_scales.std():.6f}")
        
        # 스케일 분포
        print(f"\nScale Distribution Histogram (Object PC):")
        bins = [0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, float('inf')]
        for i in range(len(bins) - 1):
            count = ((object_scales >= bins[i]) & (object_scales < bins[i+1])).sum()
            pct = 100 * count / len(object_scales)
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:5d} ({pct:5.1f}%)")

if __name__ == '__main__':
    analyze_densehograspnet_scales()
