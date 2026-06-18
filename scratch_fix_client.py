from pathlib import Path
p = Path("tests/test_ingest_cache.py")
content = p.read_text()

content = content.replace("""        for img_id in image_ids:
            for d in self.data[self.seq_id]:
                if str(d["id"]) == img_id or d["image_id"] == img_id:
                    records.append(d)""",
"""        for img_id in image_ids:
            for d in self.data[self.seq_id]:
                if d["image_id"] == img_id:
                    # Construct raw API format
                    records.append({
                        "id": img_id,
                        "sequence_id": self.seq_id,
                        "geometry": {"coordinates": [d["lon"], d["lat"], 3.0]},
                        "captured_at": d["captured_at_ms"],
                        "camera_type": d["camera_type"],
                        "camera_parameters": d.get("camera_parameters", []),
                        "quality_score": d.get("quality_score", 0.9),
                        "thumb_1024_url": d.get("thumbnail_url"),
                    })""")
p.write_text(content)
