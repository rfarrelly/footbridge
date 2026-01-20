import os
import sys

import pandas as pd


def replace_team_names_in_folder(
    folder_path, old_name, new_name, columns=["Home", "Away"]
):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(full_path)

                    modified = False
                    for col in columns:
                        if col in df.columns:
                            count = df[col].isin([old_name]).sum()
                            if count > 0:
                                df[col] = df[col].replace(old_name, new_name)
                                modified = True

                    if modified:
                        df.to_csv(full_path, index=False)
                        print(f"✅ Updated: {full_path}")
                except Exception as e:
                    print(f"❌ Failed: {full_path} – {e}")


def main():
    if len(sys.argv) != 4:
        print("Usage:")
        print(
            "  python replace_team_name.py <folder_path> <old_team_name> <new_team_name>"
        )
        sys.exit(1)

    folder_path = sys.argv[1]
    old_name = sys.argv[2]
    new_name = sys.argv[3]

    if not os.path.isdir(folder_path):
        print(f"Error: Folder does not exist: {folder_path}")
        sys.exit(1)

    print(f"Replacing '{old_name}' with '{new_name}' in folder: {folder_path}")
    replace_team_names_in_folder(folder_path, old_name, new_name)


if __name__ == "__main__":
    main()
