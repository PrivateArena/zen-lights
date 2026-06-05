import os
import sys
import json
import shutil
import xml.etree.ElementTree as ET

def validate_svg(filepath):
    """
    Validates that the file exists, is valid XML, and starts with an <svg> tag.
    Returns True if valid, False otherwise.
    """
    try:
        if not os.path.exists(filepath):
            return False
        # Try to parse it as XML to ensure it's not malformed
        tree = ET.parse(filepath)
        root = tree.getroot()
        # Ensure the root tag is svg (ignoring namespaces)
        if not root.tag.endswith('svg'):
            return False
        return True
    except Exception as e:
        print(f"Skipping malformed SVG {filepath}: {e}")
        return False

def main():
    workspace = "/media/jang/home/Deve/zen-lights"
    temp_dir = os.path.join(workspace, "assets/svg/temp/package")
    dest_dir = os.path.join(workspace, "assets/svg/data/lucide")
    registry_path = os.path.join(workspace, "assets/svg/registry.json")
    
    tags_file = os.path.join(temp_dir, "tags.json")
    icons_dir = os.path.join(temp_dir, "icons")
    
    if not os.path.exists(tags_file) or not os.path.exists(icons_dir):
        print("Error: Lucide package structure not found in temp dir.")
        sys.exit(1)
        
    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)
    
    # Load original tags
    with open(tags_file, "r") as f:
        tags_data = json.load(f)
        
    registry = []
    skipped_count = 0
    copied_count = 0
    
    # Process all SVG files
    for filename in os.listdir(icons_dir):
        if not filename.endswith(".svg"):
            continue
            
        src_path = os.path.join(icons_dir, filename)
        icon_name = filename[:-4] # strip .svg
        
        # Validate SVG
        if not validate_svg(src_path):
            skipped_count += 1
            continue
            
        # Copy to standardized folder
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(src_path, dest_path)
        copied_count += 1
        
        # Get tags from tags.json, default to empty list
        tags = tags_data.get(icon_name, [])
        # Also add the name of the icon as a tag for direct match
        if icon_name not in tags:
            tags.append(icon_name)
            
        registry.append({
            "name": icon_name,
            "dataset": "lucide",
            "filename": filename,
            "tags": tags
        })
        
    # Save the registry.json
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
        
    print(f"Build complete. Copied: {copied_count}, Skipped: {skipped_count}.")
    print(f"Registry written to {registry_path}")
    
    # Cleanup temp dir
    temp_root = os.path.join(workspace, "assets/svg/temp")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
        print("Cleaned up temporary assets.")

if __name__ == "__main__":
    main()
