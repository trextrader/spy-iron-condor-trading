#!/usr/bin/env python3
import os

def convert_py_to_txt(src_folder: str, dest_folder: str):
    # Ensure destination folder exists
    os.makedirs(dest_folder, exist_ok=True)

    for filename in os.listdir(src_folder):
        if filename.endswith(".py"):
            py_path = os.path.join(src_folder, filename)
            txt_filename = filename.replace(".py", ".txt")
            txt_path = os.path.join(dest_folder, txt_filename)

            with open(py_path, "r", encoding="utf-8") as f_in:
                content = f_in.read()

            with open(txt_path, "w", encoding="utf-8") as f_out:
                f_out.write(content)

            print(f"Converted {filename} -> {txt_path}")

if __name__ == "__main__":
    # Current working directory
    current_folder = os.getcwd()
    dest_folder = os.path.join(current_folder, "py2txt")
    convert_py_to_txt(current_folder, dest_folder)
