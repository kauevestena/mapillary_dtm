import sys
import os
import shutil
from pathlib import Path
import numpy as np
import h5py
import cv2
import imageio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tests.sample_loader import get_sample_frames
from dtm_from_mapillary.geom.sfm_dim import run as run_dim
import dtm_from_mapillary.constants as constants

def draw_matches(img1, kp1, img2, kp2, matches, out_path, num_matches=None):
    if num_matches is not None and len(matches) > num_matches:
        matches = matches[:num_matches]

    # Convert kps to cv2 KeyPoint objects
    cv_kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1.0) for pt in kp1]
    cv_kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1.0) for pt in kp2]

    # Create cv2 DMatch objects
    cv_matches = [cv2.DMatch(_queryIdx=int(m[0]), _trainIdx=int(m[1]), _imgIdx=0, _distance=0.0) for m in matches]

    out_img = cv2.drawMatches(
        img1, cv_kp1,
        img2, cv_kp2,
        cv_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(out_path), out_img)

def build_feature_tracks(matches_dict, num_images, image_names):
    tracks = []
    if num_images < 2:
        return tracks
        
    name0 = image_names[0]
    name1 = image_names[1]
    pair = (name0, name1) if (name0, name1) in matches_dict else (name1, name0)
    
    if pair not in matches_dict:
        return tracks
        
    m01 = matches_dict[pair]
    if pair[0] == name1: 
        m01 = m01[:, ::-1]

    for m in m01:
        tracks.append({0: m[0], 1: m[1]})

    for i in range(1, num_images - 1):
        nameA = image_names[i]
        nameB = image_names[i+1]
        pair = (nameA, nameB) if (nameA, nameB) in matches_dict else (nameB, nameA)
        if pair not in matches_dict:
            break
            
        mAB = matches_dict[pair]
        if pair[0] == nameB:
            mAB = mAB[:, ::-1]
            
        lookup = {m[0]: m[1] for m in mAB}
        
        for track in tracks:
            if i in track:
                idxA = track[i]
                if idxA in lookup:
                    track[i+1] = lookup[idxA]
                    
    return [t for t in tracks if len(t) >= 3]

def generate():
    dataset_dir = ROOT / "qa" / "data" / "sample_dataset"
    matching_dir = dataset_dir / "matching"
    
    if matching_dir.exists():
        shutil.rmtree(matching_dir)
        
    matching_dir.mkdir(parents=True, exist_ok=True)
    
    seqs, _ = get_sample_frames()
    if not seqs:
        print("No sequences loaded.")
        return

    # To ensure DIM executes exactly as in the pipeline, we call `run_dim` 
    # instead of creating parallel Deep Image Matching objects here.
    constants.DIM_EXTRACTOR = "superpoint"
    constants.DIM_MATCHER = "lightglue"
    
    print("Running DIM pipeline stage...")
    results = run_dim(
        sequences=seqs,
        imagery_root=dataset_dir,
        workspace_root=matching_dir,
        progress=True
    )
    print(f"DIM pipeline stage completed. Results: {results}")

    # For each sequence, DIM runner creates workspace_root / seq_id
    for seq_id, frames in seqs.items():
        seq_workspace = matching_dir / seq_id
        feature_path = seq_workspace / "features.h5"
        match_path = seq_workspace / "matches.h5"
        
        if not feature_path.exists() or not match_path.exists():
            print(f"DIM output missing for {seq_id}")
            continue

        print(f"Extracting matches for visualization for sequence {seq_id}...")
        features = {}
        with h5py.File(feature_path, "r") as f:
            for name in f.keys():
                features[name] = np.array(f[name]["keypoints"])
                
        matches_dict = {}
        with h5py.File(match_path, "r") as f:
            for nameA in f.keys():
                group = f[nameA]
                for nameB in group.keys():
                    matches_dict[(nameA, nameB)] = np.array(group[nameB])

        # Find the staged images directory that DIM runner generated or use original?
        # DIMRunner renames images to 000000_image_id.jpg and puts them in seq_id_images!
        # Oh wait, DIMRunner deletes the seq_id_images after it finishes!
        # "if images_dir.exists(): shutil.rmtree(images_dir)" in dim_adapter.py
        
        # We need the original images and we need to match them to the renamed files in features.h5
        # The DIMRunner renames to f"{staged:06d}_{frame.image_id}{src.suffix.lower()}"
        frames_list = sorted(list(frames), key=lambda f: f.captured_at_ms)
        original_images = {}
        
        from dtm_from_mapillary.geom.colmap_adapter import _find_cached_image
        
        # Re-construct the mapping from frame.image_id to the name DIM used
        img_names = []
        for i, frame in enumerate(frames_list):
            src = _find_cached_image(frame, dataset_dir)
            if src:
                dim_name = f"{i:06d}_{frame.image_id}{src.suffix.lower()}"
                img_names.append(dim_name)
                original_images[dim_name] = src

        pairwise_dir = seq_workspace / "pairwise"
        pairwise_dir.mkdir(parents=True, exist_ok=True)

        print("Generating Pairwise Visualizations...")
        for (nameA, nameB), m in matches_dict.items():
            if nameA not in original_images or nameB not in original_images:
                continue
                
            imgA = cv2.imread(str(original_images[nameA]))
            imgB = cv2.imread(str(original_images[nameB]))
            
            kpA = features[nameA]
            kpB = features[nameB]
            
            out_all = pairwise_dir / f"{nameA}_{nameB}_all.png"
            draw_matches(imgA, kpA, imgB, kpB, m, out_all)
            
            out_50 = pairwise_dir / f"{nameA}_{nameB}_top50.png"
            draw_matches(imgA, kpA, imgB, kpB, m, out_50, num_matches=50)

        print("Generating Multi-Image Animation...")
        tracks = build_feature_tracks(matches_dict, len(img_names), img_names)
        print(f"Found {len(tracks)} tracks spanning >= 3 images.")
        
        tracks_50 = tracks[:50]
        
        def create_gif(track_list, suffix):
            frames_gif = []
            import matplotlib.cm as cm
            colors = (cm.rainbow(np.linspace(0, 1, len(track_list)))[:, :3] * 255).astype(int)
            
            for i, name in enumerate(img_names):
                if name not in original_images:
                    continue
                img = cv2.imread(str(original_images[name]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                kps = features[name]
                
                for t_idx, track in enumerate(track_list):
                    if i in track:
                        kp_idx = track[i]
                        pt = kps[kp_idx]
                        color = tuple(int(c) for c in colors[t_idx])
                        cv2.circle(img, (int(pt[0]), int(pt[1])), 8, color, -1)
                        cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (255,255,255), 2)
                        
                        if i - 1 in track:
                            prev_name = img_names[i-1]
                            prev_kps = features[prev_name]
                            prev_pt = prev_kps[track[i-1]]
                            cv2.line(img, (int(prev_pt[0]), int(prev_pt[1])), (int(pt[0]), int(pt[1])), color, 3)

                frames_gif.append(img)
                
            gif_path = seq_workspace / f"multi_match_{suffix}.gif"
            imageio.mimsave(str(gif_path), frames_gif, fps=1)

        create_gif(tracks, "all")
        create_gif(tracks_50, "top50")

    print("Validation Visualizations complete!")

if __name__ == "__main__":
    generate()
