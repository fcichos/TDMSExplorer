#!/usr/bin/env python3
"""
Example usage of TDMS Explorer module

This script demonstrates various ways to use the TDMS Explorer module
to work with TDMS files containing image data.
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, '.')

from tdms_explorer import TDMSFileExplorer, list_tdms_files, create_animation_from_tdms

def main():
    print("TDMS Explorer Module - Example Usage")
    print("=" * 50)
    
    # List available TDMS files
    print("\nAvailable TDMS files:")
    tdms_files = list_tdms_files()
    
    if not tdms_files:
        print("No TDMS files found in current directory!")
        return
    
    for i, file in enumerate(tdms_files, 1):
        print(f"  {i}. {os.path.basename(file)}")
    
    # Let user select a file or use the first one
    if len(tdms_files) > 1:
        try:
            choice = int(input(f"\nSelect a file (1-{len(tdms_files)}): ")) - 1
            if choice < 0 or choice >= len(tdms_files):
                print("Invalid choice, using first file.")
                choice = 0
        except ValueError:
            choice = 0
    else:
        choice = 0
    
    filename = tdms_files[choice]
    print(f"\nWorking with: {os.path.basename(filename)}")
    
    # Create explorer instance
    print("\nCreating TDMSFileExplorer...")
    explorer = TDMSFileExplorer(filename)
    
    # Display file contents
    print("\nFile contents:")
    explorer.print_contents()
    
    # Check if file has image data
    if not explorer.has_image_data():
        print("\nThis file does not contain image data.")
        return
    
    # Example 1: Display information about the first image
    print("\n" + "="*50)
    print("EXAMPLE 1: Get information about first image")
    print("="*50)
    
    first_image = explorer.get_image_data(0)
    if first_image is not None:
        print(f"First image shape: {first_image.shape}")
        print(f"First image data type: {first_image.dtype}")
        print(f"First image min/max values: {first_image.min():.2f} / {first_image.max():.2f}")
    
    # Example 2: Write a single image
    print("\n" + "="*50)
    print("EXAMPLE 2: Write single image")
    print("="*50)
    
    output_image = "example_single_image.png"
    explorer.write_image(0, output_image, overwrite=True)
    print(f"Wrote single image to: {output_image}")
    
    # Example 3: Write a series of images
    print("\n" + "="*50)
    print("EXAMPLE 3: Write series of images")
    print("="*50)
    
    output_dir = "example_images"
    num_images = min(10, explorer.extract_images().shape[0])  # Write first 10 images
    explorer.write_images(output_dir, start_frame=0, end_frame=num_images-1)
    print(f"Wrote {num_images} images to directory: {output_dir}")
    
    # Example 4: Display an image (interactive)
    print("\n" + "="*50)
    print("EXAMPLE 4: Display an image")
    print("="*50)
    
    try:
        image_num = int(input("Enter image number to display (0 for first): ") or "0")
        explorer.display_image(image_num)
    except ValueError:
        print("Invalid input, displaying first image.")
        explorer.display_image(0)
    except Exception as e:
        print(f"Error displaying image: {e}")
    
    # Example 5: Display animation (interactive)
    print("\n" + "="*50)
    print("EXAMPLE 5: Display animation")
    print("="*50)
    
    try:
        show_animation = input("Show animation? (y/n): ").lower()
        if show_animation.startswith('y'):
            fps = int(input("Enter frames per second (default 10): ") or "10")
            explorer.display_animation(fps=fps)
    except Exception as e:
        print(f"Error displaying animation: {e}")
    
    # Example 6: Create and save animation
    print("\n" + "="*50)
    print("EXAMPLE 6: Create and save animation")
    print("="*50)
    
    try:
        create_animation = input("Create and save animation? (y/n): ").lower()
        if create_animation.startswith('y'):
            animation_file = input("Enter output filename (e.g., 'animation.mp4'): ") or "example_animation.mp4"
            fps = int(input("Enter frames per second (default 10): ") or "10")
            create_animation_from_tdms(filename, animation_file, fps=fps)
            print(f"Animation saved to: {animation_file}")
    except Exception as e:
        print(f"Error creating animation: {e}")
    
    print("\n" + "="*50)
    print("Example usage completed!")
    print("="*50)

if __name__ == "__main__":
    main()