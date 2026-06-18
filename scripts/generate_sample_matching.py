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

import deep_image_matching as dim

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
    # This builds simple tracks across the sequential images
    # matches_dict keys: (img_name_A, img_name_B) -> array of shape (M, 2)
    tracks = []
    
    # Simple tracking: point in img 0 -> img 1 -> img 2 -> img 3
    if num_images < 2:
        return tracks
        
    name0 = image_names[0]
    name1 = image_names[1]
    pair = (name0, name1) if (name0, name1) in matches_dict else (name1, name0)
    
    if pair not in matches_dict:
        return tracks
        
    m01 = matches_dict[pair]
    if pair[0] == name1: # swap if reversed
        m01 = m01[:, ::-1]

    # start a track for each match
    # track is a dict mapping image index to keypoint index
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
            
        # create a lookup from idxA to idxB
        lookup = {m[0]: m[1] for m in mAB}
        
        # update tracks
        for track in tracks:
            if i in track:
                idxA = track[i]
                if idxA in lookup:
                    track[i+1] = lookup[idxA]
                    
    return [t for t in tracks if len(t) >= 3] # only keep tracks spanning at least 3 images

def generate():
    dataset_dir = ROOT / "qa" / "data" / "sample_dataset"
    imagery_dir = dataset_dir / "imagery"
    matching_dir = dataset_dir / "matching"
    pairwise_dir = matching_dir / "pairwise"
    
    if matching_dir.exists():
        shutil.rmtree(matching_dir)
        
    matching_dir.mkdir(parents=True, exist_ok=True)
    # The imagery is structured as imagery/<seq_id>/*.jpg
    # Deep Image Matching expects images to be directly in the provided folder (not nested)
    seq_dirs = [d for d in imagery_dir.iterdir() if d.is_dir()]
    if not seq_dirs:
        print("No sequence directories found.")
        return
        
    seq_dir = seq_dirs[0]
    
    img_paths = sorted(list(seq_dir.glob("*.jpg")))
    img_names = [p.name for p in img_paths]
    
    if len(img_paths) < 2:
        print("Not enough images for matching.")
        return

    db_path = matching_dir / "database.db"

    print("Running Deep Image Matching...")
    args = {
        "images": seq_dir,
        "outs": matching_dir,
        "pipeline": "superpoint+lightglue",
        "strategy": "sequential",
        "overlap": 2,
        "quality": "high",
        "tiling": "none",
        "force": True,
    }
    
    dim_cfg = dim.Config(args)
    matcher = dim.ImageMatcher(dim_cfg)
    feature_path, match_path = matcher.run()
    
    print("Exporting to COLMAP...")
    dim.io.export_to_colmap(
        img_dir=seq_dir,
        feature_path=feature_path,
        match_path=match_path,
        database_path=str(db_path),
        camera_config_path=dim_cfg.general["camera_options"],
    )
    
    print("Extracting matches for visualization...")
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

    pairwise_dir = matching_dir / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Pairwise Visualizations...")
    for (nameA, nameB), m in matches_dict.items():
        imgA_path = seq_dir / nameA
        imgB_path = seq_dir / nameB
        
        if not imgA_path.exists() or not imgB_path.exists():
            continue
            
        imgA = cv2.imread(str(imgA_path))
        imgB = cv2.imread(str(imgB_path))
        
        kpA = features[nameA]
        kpB = features[nameB]
        
        # Generate 'all matches'
        out_all = pairwise_dir / f"{nameA}_{nameB}_all.png"
        draw_matches(imgA, kpA, imgB, kpB, m, out_all)
        
        # Generate 'top 50' matches (since lightglue may not return scores in a simple way here, we just take 50 random or first 50)
        # Note: In matches.h5 from dim, typically matches are sorted by confidence if lightglue is used. 
        # We will just take the first 50 which are often the highest confidence or random enough to be clean.
        out_50 = pairwise_dir / f"{nameA}_{nameB}_top50.png"
        draw_matches(imgA, kpA, imgB, kpB, m, out_50, num_matches=50)

    print("Generating Multi-Image Animation...")
    tracks = build_feature_tracks(matches_dict, len(img_names), img_names)
    print(f"Found {len(tracks)} tracks spanning >= 3 images.")
    
    # Top 50 tracks
    tracks_50 = tracks[:50]
    
    def create_gif(track_list, suffix):
        frames = []
        import matplotlib.cm as cm
        colors = (cm.rainbow(np.linspace(0, 1, len(track_list)))[:, :3] * 255).astype(int)
        
        for i, name in enumerate(img_names):
            img_path = seq_dir / name
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            kps = features[name]
            
            for t_idx, track in enumerate(track_list):
                if i in track:
                    kp_idx = track[i]
                    pt = kps[kp_idx]
                    color = tuple(int(c) for c in colors[t_idx])
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 8, color, -1)
                    cv2.circle(img, (int(pt[0]), int(pt[1])), 10, (255,255,255), 2)
                    
                    # Draw a trail from previous frame
                    if i - 1 in track:
                        prev_name = img_names[i-1]
                        prev_kps = features[prev_name]
                        prev_pt = prev_kps[track[i-1]]
                        cv2.line(img, (int(prev_pt[0]), int(prev_pt[1])), (int(pt[0]), int(pt[1])), color, 3)

            frames.append(img)
            
        gif_path = matching_dir / f"multi_match_{suffix}.gif"
        imageio.mimsave(str(gif_path), frames, fps=1)

    create_gif(tracks, "all")
    create_gif(tracks_50, "top50")

    print("Validation Visualizations complete!")

if __name__ == "__main__":
    generate()
