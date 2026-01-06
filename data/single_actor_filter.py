import os
import re

def filter_single_actor_video(folder_path: str):
    # Conters for tracking
    deleted_count = 0
    kept_count = 0

    # Pattern to find "A" followed by 3 digits
    action_pattern = re.compile(r'A(\d{3})')

    print(f"Scanning folder: {folder_path}...")

    for filename in os.listdir(folder_path):
        if not filename.endswith('.avi'):
            continue

        match = action_pattern.search(filename)
        if match: 
            action_id = int(match.group(1))      # Extract the numeric part

            # Define two-actor action ranges
            # NTU 60: A050 - A060
            # NTU 120: A106 - A120
            is_two_actor = (50 <= action_id <= 60) or (106 <= action_id <= 120)

            if is_two_actor:
                file_to_del = os.path.join(folder_path, filename)
                try:
                    os.remove(file_to_del)
                    deleted_count += 1
                    print(f"Deleted two-actor video: {filename}")
                except Exception as e:
                    print(f"Error deleting file {filename}: {e}")
            else:
                kept_count += 1

    print("-"*30)
    print(f"Filtering Complete!")
    print(f"Videos kept (1 actor): {kept_count}")
    print(f"Videos deleted (2 actors): {deleted_count}")

    # Rename parent folder
    parent_folder = os.path.dirname(folder_path)
    grandparent_folder = os.path.dirname(parent_folder)
    parent_folder_name = os.path.basename(parent_folder)

    # Pattern to match nturgbd_rgb_s0xx
    rename_pattern = re.compile(r'^(nturgbd_rgb_s\d{3})$')
    match = rename_pattern.match(parent_folder_name)

    if match:
        new_parent_folder_name = f"{parent_folder_name}_single_actor"
        new_parent_folder = os.path.join(grandparent_folder, new_parent_folder_name)

        try:
            os.rename(parent_folder, new_parent_folder)
            print(f"Renamed folder: {parent_folder_name} -> {new_parent_folder_name}")
        except Exception as e:
            print(f"Error renaming folder: {e}")
    else:
        print(f"Folder name '{parent_folder_name}' does not match expected pattern, skipping rename.")

if __name__ == "__main__":
    for i in range(4, 11):
        index = str(i).zfill(2)
        data_folder = f"/media/phonght/New Volume/NTU-dataset/rgb-videos/nturgbd_rgb_s0{index}/nturgb+d_rgb"
        filter_single_actor_video(data_folder)