import os
import pandas as pd

def make_csv_from_folders(root_dir, output_csv="train.csv"):
    rows = []
    for country in os.listdir(root_dir):
        country_path = os.path.join(root_dir, country)
        if os.path.isdir(country_path):  # only go into folders
            for fname in os.listdir(country_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    rows.append({
                        "image_path": f"{country}/{fname}",
                        "country_code": country
                    })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv} with {len(df)} rows.")

# Example usage:
make_csv_from_folders(r"use_example/here", "train.csv")  # insert your directory here
