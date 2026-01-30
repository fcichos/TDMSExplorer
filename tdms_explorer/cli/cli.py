"""
TDMS Explorer CLI Module

Command Line Interface for the TDMS Explorer package.
"""

import sys
import os
import argparse
import textwrap
from typing import List, Optional

try:
    from tdms_explorer.core import TDMSFileExplorer, list_tdms_files, create_animation_from_tdms
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: {e}")
    print("Please ensure all dependencies are installed.")
    sys.exit(1)


class TDMS_CLI:
    """Command Line Interface for TDMS Explorer"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.args = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all commands and options."""
        
        # Main parser
        parser = argparse.ArgumentParser(
            description='TDMS Explorer CLI - Command line tool for working with TDMS files',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent('''
Examples:
  # List all TDMS files
  python -m tdms_explorer.cli list
  
  # Show information about a specific file
  python -m tdms_explorer.cli info "file.tdms"
  
  # Display first image from a file
  python -m tdms_explorer.cli show "file.tdms" --image 0
  
  # Save image to file (works in JupyterLab!)
  python -m tdms_explorer.cli show "file.tdms" --save output.png
  
  # Export all images to directory
  python -m tdms_explorer.cli export "file.tdms" output_dir
  
  # Create animation
  python -m tdms_explorer.cli animate "file.tdms" animation.mp4 --fps 10
  
  # Get raw data info
  python -m tdms_explorer.cli raw "file.tdms" --group "Image" --channel "Image"
  
  # Image Analysis Examples:
  
  # Analyze image statistics
  python -m tdms_explorer.cli analyze "file.tdms" --image 0
  
  # Apply Gaussian filter
  python -m tdms_explorer.cli filter "file.tdms" --image 0 --type gaussian --sigma 1.5
  
  # Detect edges using Canny method
  python -m tdms_explorer.cli edges "file.tdms" --image 0 --method canny
  
  # Apply Otsu thresholding
  python -m tdms_explorer.cli threshold "file.tdms" --image 0 --method otsu
  
  # Set and analyze ROI
  python -m tdms_explorer.cli roi "file.tdms" --image 0 --set --x 50 --y 50 --width 200 --height 200
  python -m tdms_explorer.cli roi "file.tdms" --image 0 --analyze
  
  # Interactive ROI selection
  python -m tdms_explorer.cli roi "file.tdms" --image 0 --interactive
  
  # Compare two images
  python -m tdms_explorer.cli compare "file.tdms" --image1 0 --image2 1
  
  # Show image histogram
  python -m tdms_explorer.cli histogram "file.tdms" --image 0
  
  # Detect features (blobs, corners, edges)
  python -m tdms_explorer.cli features "file.tdms" --image 0 --method blob
  
  # Get intensity profile
  python -m tdms_explorer.cli profile "file.tdms" --image 0 --direction horizontal
            ''')
        )
        
        # Subparsers for different commands
        subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)
        
        # List command
        list_parser = subparsers.add_parser('list', help='List all TDMS files in directory')
        list_parser.add_argument('--dir', '-d', default='.', help='Directory to search (default: current directory)')
        list_parser.add_argument('--details', '-l', action='store_true', help='Show detailed information')
        
        # Info command
        info_parser = subparsers.add_parser('info', help='Show detailed information about a TDMS file')
        info_parser.add_argument('file', help='TDMS file to analyze')
        info_parser.add_argument('--json', action='store_true', help='Output in JSON format')
        
        # Show command
        show_parser = subparsers.add_parser('show', help='Display images from TDMS file')
        show_parser.add_argument('file', help='TDMS file to display')
        show_parser.add_argument('--image', '-i', type=int, default=0, help='Image number to display (default: 0)')
        show_parser.add_argument('--cmap', default='gray', help='Colormap to use (default: gray)')
        show_parser.add_argument('--no-show', action='store_true', help='Don\'t display image (show statistics instead)')
        show_parser.add_argument('--save', help='Save image to file instead of displaying')
        
        # Animate command
        animate_parser = subparsers.add_parser('animate', help='Create animation from TDMS file')
        animate_parser.add_argument('file', help='TDMS file to animate')
        animate_parser.add_argument('output', help='Output animation file (e.g., animation.mp4)')
        animate_parser.add_argument('--fps', type=int, default=10, help='Frames per second (default: 10)')
        animate_parser.add_argument('--cmap', default='gray', help='Colormap to use (default: gray)')
        animate_parser.add_argument('--start', type=int, default=0, help='Start frame (default: 0)')
        animate_parser.add_argument('--end', type=int, help='End frame (default: last frame)')
        animate_parser.add_argument('--no-display', action='store_true', help='Don\'t display animation')
        
        # Export command
        export_parser = subparsers.add_parser('export', help='Export images from TDMS file')
        export_parser.add_argument('file', help='TDMS file to export')
        export_parser.add_argument('output_dir', help='Output directory for images')
        export_parser.add_argument('--start', type=int, default=0, help='Start frame (default: 0)')
        export_parser.add_argument('--end', type=int, help='End frame (default: last frame)')
        export_parser.add_argument('--prefix', default='output_', help='Filename prefix (default: output_)')
        export_parser.add_argument('--format', default='png', help='Image format (default: png)')
        export_parser.add_argument('--cmap', default='gray', help='Colormap to use (default: gray)')
        export_parser.add_argument('--single', type=int, help='Export only single image number')
        export_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
        
        # Raw command
        raw_parser = subparsers.add_parser('raw', help='Access raw channel data')
        raw_parser.add_argument('file', help='TDMS file to read')
        raw_parser.add_argument('--group', '-g', required=True, help='Group name')
        raw_parser.add_argument('--channel', '-c', required=True, help='Channel name')
        raw_parser.add_argument('--info', action='store_true', help='Show channel info only')
        raw_parser.add_argument('--save', help='Save raw data to file (numpy format)')
        
        # Stats command
        stats_parser = subparsers.add_parser('stats', help='Show statistics about TDMS file')
        stats_parser.add_argument('file', help='TDMS file to analyze')
        stats_parser.add_argument('--images', action='store_true', help='Show image statistics')
        stats_parser.add_argument('--channels', action='store_true', help='Show channel statistics')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze images from TDMS file')
        analyze_parser.add_argument('file', help='TDMS file to analyze')
        analyze_parser.add_argument('--image', '-i', type=int, default=0, help='Image number to analyze (default: 0)')
        analyze_parser.add_argument('--roi', action='store_true', help='Analyze ROI instead of full image')
        analyze_parser.add_argument('--json', action='store_true', help='Output in JSON format')
        
        # Filter command
        filter_parser = subparsers.add_parser('filter', help='Apply filters to images')
        filter_parser.add_argument('file', help='TDMS file to process')
        filter_parser.add_argument('--image', '-i', type=int, default=0, help='Image number to filter (default: 0)')
        filter_parser.add_argument('--type', '-t', choices=['gaussian', 'median', 'bilateral', 'sobel', 'prewitt', 'laplace'], 
                                  default='gaussian', help='Filter type (default: gaussian)')
        filter_parser.add_argument('--sigma', type=float, default=1.0, help='Sigma for Gaussian filter (default: 1.0)')
        filter_parser.add_argument('--size', type=int, default=3, help='Kernel size for median filter (default: 3)')
        filter_parser.add_argument('--save', help='Save filtered image to file')
        filter_parser.add_argument('--show', action='store_true', help='Show filtered image')
        
        # Edge command
        edge_parser = subparsers.add_parser('edges', help='Detect edges in images')
        edge_parser.add_argument('file', help='TDMS file to process')
        edge_parser.add_argument('--image', '-i', type=int, default=0, help='Image number to process (default: 0)')
        edge_parser.add_argument('--method', '-m', choices=['canny', 'sobel', 'prewitt', 'laplace'], 
                                default='canny', help='Edge detection method (default: canny)')
        edge_parser.add_argument('--save', help='Save edge-detected image to file')
        edge_parser.add_argument('--show', action='store_true', help='Show edge-detected image')
        
        # Threshold command
        threshold_parser = subparsers.add_parser('threshold', help='Apply thresholding to images')
        threshold_parser.add_argument('file', help='TDMS file to process')
        threshold_parser.add_argument('--image', '-i', type=int, default=0, help='Image number to process (default: 0)')
        threshold_parser.add_argument('--method', '-m', choices=['otsu', 'adaptive', 'manual'], 
                                     default='otsu', help='Thresholding method (default: otsu)')
        threshold_parser.add_argument('--threshold', type=float, help='Manual threshold value')
        threshold_parser.add_argument('--save', help='Save thresholded image to file')
        threshold_parser.add_argument('--show', action='store_true', help='Show thresholded image')
        
        # ROI command
        roi_parser = subparsers.add_parser('roi', help='Region of Interest operations')
        roi_parser.add_argument('file', help='TDMS file to process')
        roi_parser.add_argument('--image', '-i', type=int, default=0, help='Image number (default: 0)')
        roi_parser.add_argument('--set', action='store_true', help='Set ROI coordinates')
        roi_parser.add_argument('--x', type=int, default=0, help='ROI x coordinate (default: 0)')
        roi_parser.add_argument('--y', type=int, default=0, help='ROI y coordinate (default: 0)')
        roi_parser.add_argument('--width', type=int, default=100, help='ROI width (default: 100)')
        roi_parser.add_argument('--height', type=int, default=100, help='ROI height (default: 100)')
        roi_parser.add_argument('--analyze', action='store_true', help='Analyze current ROI')
        roi_parser.add_argument('--interactive', action='store_true', help='Interactive ROI selection')
        roi_parser.add_argument('--clear', action='store_true', help='Clear current ROI')
        
        # Compare command
        compare_parser = subparsers.add_parser('compare', help='Compare two images')
        compare_parser.add_argument('file', help='TDMS file to process')
        compare_parser.add_argument('--image1', '-i1', type=int, default=0, help='First image number (default: 0)')
        compare_parser.add_argument('--image2', '-i2', type=int, default=1, help='Second image number (default: 1)')
        compare_parser.add_argument('--method', '-m', choices=['difference', 'absolute', 'relative'], 
                                    default='difference', help='Comparison method (default: difference)')
        compare_parser.add_argument('--save', help='Save comparison result to file')
        compare_parser.add_argument('--show', action='store_true', help='Show comparison result')
        
        # Histogram command
        histogram_parser = subparsers.add_parser('histogram', help='Show image histogram')
        histogram_parser.add_argument('file', help='TDMS file to process')
        histogram_parser.add_argument('--image', '-i', type=int, default=0, help='Image number (default: 0)')
        histogram_parser.add_argument('--bins', type=int, default=256, help='Number of histogram bins (default: 256)')
        histogram_parser.add_argument('--log', action='store_true', help='Use logarithmic scale')
        histogram_parser.add_argument('--save', help='Save histogram data to JSON file')
        
        # Features command
        features_parser = subparsers.add_parser('features', help='Detect features in images')
        features_parser.add_argument('file', help='TDMS file to process')
        features_parser.add_argument('--image', '-i', type=int, default=0, help='Image number (default: 0)')
        features_parser.add_argument('--method', '-m', choices=['blob', 'corner', 'edge'], 
                                     default='blob', help='Feature detection method (default: blob)')
        features_parser.add_argument('--json', action='store_true', help='Output in JSON format')
        
        # Profile command
        profile_parser = subparsers.add_parser('profile', help='Get image intensity profile')
        profile_parser.add_argument('file', help='TDMS file to process')
        profile_parser.add_argument('--image', '-i', type=int, default=0, help='Image number (default: 0)')
        profile_parser.add_argument('--direction', '-d', choices=['horizontal', 'vertical', 'diagonal'], 
                                    default='horizontal', help='Profile direction (default: horizontal)')
        profile_parser.add_argument('--position', '-p', type=int, help='Position for line profile')
        profile_parser.add_argument('--save', help='Save profile data to JSON file')
        
        return parser
    
    def parse_args(self, args=None):
        """Parse command line arguments."""
        if args is None:
            args = sys.argv[1:]
        self.args = self.parser.parse_args(args)
        return self.args
    
    def run(self):
        """Run the CLI command."""
        try:
            if self.args.command == 'list':
                self._command_list()
            elif self.args.command == 'info':
                self._command_info()
            elif self.args.command == 'show':
                self._command_show()
            elif self.args.command == 'animate':
                self._command_animate()
            elif self.args.command == 'export':
                self._command_export()
            elif self.args.command == 'raw':
                self._command_raw()
            elif self.args.command == 'stats':
                self._command_stats()
            elif self.args.command == 'analyze':
                self._command_analyze()
            elif self.args.command == 'filter':
                self._command_filter()
            elif self.args.command == 'edges':
                self._command_edges()
            elif self.args.command == 'threshold':
                self._command_threshold()
            elif self.args.command == 'roi':
                self._command_roi()
            elif self.args.command == 'compare':
                self._command_compare()
            elif self.args.command == 'histogram':
                self._command_histogram()
            elif self.args.command == 'features':
                self._command_features()
            elif self.args.command == 'profile':
                self._command_profile()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if hasattr(e, '__traceback__'):
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    def _command_list(self):
        """List TDMS files command."""
        directory = self.args.dir
        details = self.args.details
        
        print(f"Listing TDMS files in: {os.path.abspath(directory)}")
        print("-" * 60)
        
        files = list_tdms_files(directory)
        
        if not files:
            print("No TDMS files found.")
            return
        
        for i, file in enumerate(files, 1):
            size = os.path.getsize(file)
            base_name = os.path.basename(file)
            
            if details:
                try:
                    explorer = TDMSFileExplorer(file)
                    has_images = explorer.has_image_data()
                    groups = len(explorer.groups)
                    print(f"{i:2d}. {base_name}")
                    print(f"    Size: {size:,} bytes")
                    print(f"    Groups: {groups}")
                    print(f"    Has images: {has_images}")
                except Exception as e:
                    print(f"{i:2d}. {base_name} - Error: {e}")
            else:
                print(f"{i:2d}. {base_name} ({size:,} bytes)")
        
        print(f"\nFound {len(files)} TDMS file(s).")
    
    def _command_info(self):
        """Info command."""
        filename = self.args.file
        json_output = self.args.json
        
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found.")
            return
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if json_output:
                import json
                info = explorer.list_contents()
                print(json.dumps(info, indent=2))
            else:
                explorer.print_contents()
                
                # Additional useful info
                if explorer.has_image_data():
                    images = explorer.extract_images()
                    if images is not None:
                        print(f"\nImage Data:")
                        print(f"  Total frames: {images.shape[0]}")
                        print(f"  Frame size: {images.shape[1]}x{images.shape[2]}")
                        print(f"  Data type: {images.dtype}")
                        print(f"  Memory size: {images.nbytes:,} bytes")
        
        except Exception as e:
            print(f"Error analyzing file: {e}")
    
    def _command_show(self):
        """Show image command."""
        filename = self.args.file
        image_num = self.args.image
        cmap = self.args.cmap
        no_show = self.args.no_show
        save_file = self.args.save
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
            
            images = explorer.extract_images()
            if images is None:
                print("Could not extract images.")
                return
            
            if image_num >= images.shape[0]:
                print(f"Image number {image_num} is out of range. Max: {images.shape[0]-1}")
                return
            
            print(f"Image {image_num} from {filename}")
            print(f"Shape: {images.shape[1]}x{images.shape[2]}")
            print(f"Data range: {images[image_num].min():.2f} to {images[image_num].max():.2f}")
            print(f"Data type: {images.dtype}")
            
            # Handle save option first
            if save_file:
                print(f"\n💾 Saving image to: {save_file}")
                explorer.write_image(image_num, save_file, cmap=cmap, overwrite=True)
                print(f"✅ Image saved successfully!")
                
                # Also show statistics when saving
                print(f"\n📊 Image Statistics:")
                img_data = images[image_num, :, :]
                print(f"  Min: {img_data.min():.4f}")
                print(f"  Max: {img_data.max():.4f}")
                print(f"  Mean: {img_data.mean():.4f}")
                print(f"  Std: {img_data.std():.4f}")
                print(f"  Median: {np.median(img_data):.4f}")
                return
            
            # Check if we're in a headless environment
            import matplotlib
            backend = matplotlib.get_backend()
            is_headless = backend in ['Agg', 'pdf', 'ps', 'svg'] or 'inline' not in backend
            
            if no_show or is_headless:
                print("\n📊 Image Statistics:")
                img_data = images[image_num, :, :]
                print(f"  Min: {img_data.min():.4f}")
                print(f"  Max: {img_data.max():.4f}")
                print(f"  Mean: {img_data.mean():.4f}")
                print(f"  Std: {img_data.std():.4f}")
                print(f"  Median: {np.median(img_data):.4f}")
                
                # Offer to save the image
                print(f"\n💾 Tip: Use '--save image.png' to save this image to a file")
                print(f"🎨 Tip: Use '--no-show' to skip display in GUI environments")
                
                if is_headless:
                    print(f"\n⚠️  Display not available in this environment (backend: {backend})")
                    print(f"   Try running in a terminal with GUI support or use '--save' to export the image")
                else:
                    print("Image display skipped (--no-show flag)")
            else:
                try:
                    print("\n🖼️  Displaying image...")
                    explorer.display_image(image_num, cmap=cmap)
                except Exception as display_error:
                    print(f"\n❌ Could not display image: {display_error}")
                    print("   Falling back to statistics display...")
                    img_data = images[image_num, :, :]
                    print(f"\n📊 Image Statistics:")
                    print(f"  Min: {img_data.min():.4f}")
                    print(f"  Max: {img_data.max():.4f}")
                    print(f"  Mean: {img_data.mean():.4f}")
                    print(f"  Std: {img_data.std():.4f}")
                    print(f"  Median: {np.median(img_data):.4f}")
                    print(f"\n💾 Use '--save image.png' to save this image to a file")
                
        except Exception as e:
            print(f"Error displaying image: {e}")
    
    def _command_animate(self):
        """Animate command."""
        filename = self.args.file
        output = self.args.output
        fps = self.args.fps
        cmap = self.args.cmap
        start = self.args.start
        end = self.args.end
        no_display = self.args.no_display
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
            
            images = explorer.extract_images()
            if images is None:
                print("Could not extract images.")
                return
            
            if end is None:
                end = images.shape[0] - 1
            
            if start < 0 or end >= images.shape[0] or start > end:
                print(f"Invalid frame range. Available: 0 to {images.shape[0]-1}")
                return
            
            total_frames = end - start + 1
            print(f"Creating animation from {filename}")
            print(f"Frames: {start} to {end} ({total_frames} frames)")
            print(f"FPS: {fps}")
            print(f"Output: {output}")
            
            # Create animation
            create_animation_from_tdms(filename, output, fps=fps, cmap=cmap)
            
            if not no_display:
                print(f"\nAnimation created: {output}")
                print(f"File size: {os.path.getsize(output):,} bytes")
            else:
                print(f"Animation saved to: {output}")
                
        except Exception as e:
            print(f"Error creating animation: {e}")
    
    def _command_export(self):
        """Export command."""
        filename = self.args.file
        output_dir = self.args.output_dir
        start = self.args.start
        end = self.args.end
        prefix = self.args.prefix
        format = self.args.format
        cmap = self.args.cmap
        single = self.args.single
        overwrite = self.args.overwrite
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
            
            images = explorer.extract_images()
            if images is None:
                print("Could not extract images.")
                return
            
            if single is not None:
                # Export single image
                if single >= images.shape[0]:
                    print(f"Image number {single} is out of range. Max: {images.shape[0]-1}")
                    return
                
                output_path = os.path.join(output_dir, f"{prefix}{single:03d}.{format}")
                print(f"Exporting single image {single} to {output_path}")
                explorer.write_image(single, output_path, cmap=cmap, overwrite=overwrite)
                print(f"✓ Exported image {single}")
                
            else:
                # Export image series
                if end is None:
                    end = images.shape[0] - 1
                
                if start < 0 or end >= images.shape[0] or start > end:
                    print(f"Invalid frame range. Available: 0 to {images.shape[0]-1}")
                    return
                
                total_frames = end - start + 1
                print(f"Exporting {total_frames} images from {filename}")
                print(f"Frames: {start} to {end}")
                print(f"Output directory: {output_dir}")
                print(f"Format: {format}")
                print(f"Prefix: {prefix}")
                
                explorer.write_images(
                    output_dir, 
                    start_frame=start, 
                    end_frame=end, 
                    prefix=prefix, 
                    format=format, 
                    cmap=cmap
                )
                
                # Show summary
                exported_files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(f'.{format}')]
                print(f"\n✓ Exported {len(exported_files)} images to {output_dir}")
                
        except Exception as e:
            print(f"Error exporting images: {e}")
    
    def _command_raw(self):
        """Raw data command."""
        filename = self.args.file
        group_name = self.args.group
        channel_name = self.args.channel
        info_only = self.args.info
        save_file = self.args.save
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            # Check if group and channel exist
            if group_name not in explorer.groups:
                print(f"Error: Group '{group_name}' not found.")
                print(f"Available groups: {', '.join(explorer.groups)}")
                return
            
            channels = explorer.channels[group_name].get('_channels', [])
            if channel_name not in channels:
                print(f"Error: Channel '{channel_name}' not found in group '{group_name}'.")
                print(f"Available channels: {', '.join(channels)}")
                return
            
            if info_only:
                # Show channel info
                channel_info = explorer.channels[group_name][channel_name]
                print(f"Channel: {group_name}/{channel_name}")
                print(f"  Data type: {channel_info['data_type']}")
                print(f"  Shape: {channel_info['shape']}")
                print(f"  Size: {channel_info['size']:,} elements")
            else:
                # Get raw data
                data = explorer.get_raw_channel_data(group_name, channel_name)
                if data is None:
                    print("Could not read channel data.")
                    return
                
                print(f"Channel: {group_name}/{channel_name}")
                print(f"  Shape: {data.shape}")
                print(f"  Data type: {data.dtype}")
                print(f"  Size: {data.size:,} elements")
                print(f"  Memory: {data.nbytes:,} bytes")
                
                if data.size > 0:
                    print(f"  Min: {np.min(data):.6f}")
                    print(f"  Max: {np.max(data):.6f}")
                    print(f"  Mean: {np.mean(data):.6f}")
                    print(f"  Std: {np.std(data):.6f}")
                
                # Save data if requested
                if save_file:
                    try:
                        np.save(save_file, data)
                        print(f"\n✓ Saved raw data to: {save_file}")
                        print(f"  File size: {os.path.getsize(save_file):,} bytes")
                    except Exception as e:
                        print(f"Error saving data: {e}")
                
        except Exception as e:
            print(f"Error accessing raw data: {e}")
    
    def _command_stats(self):
        """Stats command."""
        filename = self.args.file
        show_images = self.args.images
        show_channels = self.args.channels
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            print(f"Statistics for: {filename}")
            print("=" * 50)
            
            # File info
            print("File Information:")
            print(f"  Size: {explorer.metadata.get('file_size', 'Unknown'):,} bytes")
            print(f"  Groups: {len(explorer.groups)}")
            print(f"  Has image data: {explorer.has_image_data()}")
            
            if show_images and explorer.has_image_data():
                print("\nImage Statistics:")
                images = explorer.extract_images()
                if images is not None:
                    print(f"  Total frames: {images.shape[0]}")
                    print(f"  Frame size: {images.shape[1]}x{images.shape[2]}")
                    print(f"  Data type: {images.dtype}")
                    print(f"  Memory usage: {images.nbytes:,} bytes")
                    
                    # Calculate statistics for first few images
                    sample_size = min(10, images.shape[0])
                    sample = images[:sample_size]
                    
                    print(f"  Sample statistics (first {sample_size} images):")
                    print(f"    Min: {sample.min():.2f}")
                    print(f"    Max: {sample.max():.2f}")
                    print(f"    Mean: {sample.mean():.2f}")
                    print(f"    Std: {sample.std():.2f}")
            
            if show_channels:
                print("\nChannel Statistics:")
                for group_name in explorer.groups:
                    print(f"  Group: {group_name}")
                    channels = explorer.channels[group_name].get('_channels', [])
                    for channel_name in channels:
                        if channel_name == '_channels':
                            continue
                        
                        channel_info = explorer.channels[group_name][channel_name]
                        print(f"    {channel_name}:")
                        print(f"      Type: {channel_info['data_type']}")
                        print(f"      Shape: {channel_info['shape']}")
                        print(f"      Size: {channel_info['size']:,} elements")
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
    
    def _command_analyze(self):
        """Analyze image command."""
        filename = self.args.file
        image_num = self.args.image
        analyze_roi = self.args.roi
        json_output = self.args.json
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            if analyze_roi:
                # Analyze ROI
                roi_coords = explorer.set_roi(image_num, 50, 50, 200, 200)
                if roi_coords:
                    print(f"🎯 ROI set at: {roi_coords}")
                    analysis = explorer.get_roi_analysis(image_num)
                else:
                    print("Failed to set ROI")
                    return
            else:
                # Analyze full image
                analysis = explorer.analyze_image(image_num)
                
            if analysis is None:
                print("Analysis failed.")
                return
                
            if json_output:
                import json
                print(json.dumps(analysis, indent=2))
            else:
                print(f"📊 Analysis Results for Image {image_num}:")
                print("=" * 50)
                print(f"Basic Statistics:")
                print(f"  Min: {analysis['min']:.4f}")
                print(f"  Max: {analysis['max']:.4f}")
                print(f"  Mean: {analysis['mean']:.4f}")
                print(f"  Std: {analysis['std']:.4f}")
                print(f"  Median: {analysis['median']:.4f}")
                
                if 'entropy' in analysis:
                    print(f"  Entropy: {analysis['entropy']:.4f}")
                
                if 'edge_density' in analysis and analysis['edge_density'] is not None:
                    print(f"  Edge Density: {analysis['edge_density']:.4f}")
                
                if 'roi' in analysis:
                    print(f"\nROI Statistics:")
                    roi_stats = analysis['roi']
                    print(f"  Area: {roi_stats['area']} pixels")
                    print(f"  Min: {roi_stats['min']:.4f}")
                    print(f"  Max: {roi_stats['max']:.4f}")
                    print(f"  Mean: {roi_stats['mean']:.4f}")
                    print(f"  Std: {roi_stats['std']:.4f}")
                    print(f"  Median: {roi_stats['median']:.4f}")
        
        except Exception as e:
            print(f"Error analyzing image: {e}")
    
    def _command_filter(self):
        """Filter image command."""
        filename = self.args.file
        image_num = self.args.image
        filter_type = self.args.type
        sigma = self.args.sigma
        size = self.args.size
        save_file = self.args.save
        show_image = self.args.show
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            print(f"🔧 Applying {filter_type} filter to image {image_num}...")
            
            # Apply filter with appropriate parameters
            if filter_type == 'gaussian':
                filtered = explorer.apply_image_filter(image_num, filter_type, sigma=sigma)
            elif filter_type == 'median':
                filtered = explorer.apply_image_filter(image_num, filter_type, size=size)
            else:
                filtered = explorer.apply_image_filter(image_num, filter_type)
                
            if filtered is None:
                print("Filter application failed.")
                return
                
            print(f"✅ Filter applied successfully")
            
            # Save if requested
            if save_file:
                print(f"💾 Saving filtered image to: {save_file}")
                plt.imsave(save_file, filtered, cmap='gray')
                print(f"✅ Image saved successfully")
                
            # Show if requested
            if show_image:
                print("🖼️  Displaying filtered image...")
                plt.figure(figsize=(10, 8))
                plt.imshow(filtered, cmap='gray')
                plt.title(f"Filtered Image ({filter_type}) - Image {image_num}")
                plt.colorbar()
                plt.show()
                
        except Exception as e:
            print(f"Error applying filter: {e}")
    
    def _command_edges(self):
        """Edge detection command."""
        filename = self.args.file
        image_num = self.args.image
        method = self.args.method
        save_file = self.args.save
        show_image = self.args.show
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            print(f"🔍 Detecting edges in image {image_num} using {method} method...")
            
            edges = explorer.detect_edges(image_num, method)
            
            if edges is None:
                print("Edge detection failed.")
                return
                
            print(f"✅ Edge detection completed")
            
            # Save if requested
            if save_file:
                print(f"💾 Saving edge-detected image to: {save_file}")
                plt.imsave(save_file, edges, cmap='gray')
                print(f"✅ Image saved successfully")
                
            # Show if requested
            if show_image:
                print("🖼️  Displaying edge-detected image...")
                plt.figure(figsize=(10, 8))
                plt.imshow(edges, cmap='gray')
                plt.title(f"Edge Detection ({method}) - Image {image_num}")
                plt.colorbar()
                plt.show()
                
        except Exception as e:
            print(f"Error detecting edges: {e}")
    
    def _command_threshold(self):
        """Threshold command."""
        filename = self.args.file
        image_num = self.args.image
        method = self.args.method
        threshold = self.args.threshold
        save_file = self.args.save
        show_image = self.args.show
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            print(f"📉 Applying {method} thresholding to image {image_num}...")
            
            # Apply thresholding
            if method == 'manual' and threshold is not None:
                thresholded = explorer.threshold_image(image_num, method, threshold=threshold)
            else:
                thresholded = explorer.threshold_image(image_num, method)
                
            if thresholded is None:
                print("Thresholding failed.")
                return
                
            print(f"✅ Thresholding completed")
            
            # Save if requested
            if save_file:
                print(f"💾 Saving thresholded image to: {save_file}")
                plt.imsave(save_file, thresholded, cmap='gray')
                print(f"✅ Image saved successfully")
                
            # Show if requested
            if show_image:
                print("🖼️  Displaying thresholded image...")
                plt.figure(figsize=(10, 8))
                plt.imshow(thresholded, cmap='gray')
                plt.title(f"Thresholded Image ({method}) - Image {image_num}")
                plt.colorbar()
                plt.show()
                
        except Exception as e:
            print(f"Error applying threshold: {e}")
    
    def _command_roi(self):
        """ROI command."""
        filename = self.args.file
        image_num = self.args.image
        set_roi = self.args.set
        x = self.args.x
        y = self.args.y
        width = self.args.width
        height = self.args.height
        analyze_roi = self.args.analyze
        interactive = self.args.interactive
        clear_roi = self.args.clear
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            if set_roi:
                print(f"🎯 Setting ROI at ({x}, {y}) with size {width}x{height}")
                roi_coords = explorer.set_roi(image_num, x, y, width, height)
                if roi_coords:
                    print(f"✅ ROI set successfully: {roi_coords}")
                else:
                    print("Failed to set ROI")
                    
            elif interactive:
                print("🖱️  Starting interactive ROI selection...")
                print("Click and drag to select a rectangular region, then release.")
                roi_coords = explorer.interactive_roi_selection(image_num)
                if roi_coords:
                    print(f"✅ ROI selected: {roi_coords}")
                else:
                    print("ROI selection cancelled or failed")
                    
            elif analyze_roi:
                print(f"📊 Analyzing current ROI for image {image_num}...")
                analysis = explorer.get_roi_analysis(image_num)
                if analysis and 'roi' in analysis:
                    roi_stats = analysis['roi']
                    print(f"ROI Analysis Results:")
                    print(f"  Area: {roi_stats['area']} pixels")
                    print(f"  Min: {roi_stats['min']:.4f}")
                    print(f"  Max: {roi_stats['max']:.4f}")
                    print(f"  Mean: {roi_stats['mean']:.4f}")
                    print(f"  Std: {roi_stats['std']:.4f}")
                    print(f"  Median: {roi_stats['median']:.4f}")
                else:
                    print("No ROI is currently set or analysis failed")
                    
            elif clear_roi:
                print("🧹 Clearing current ROI...")
                # Create analyzer and clear ROI
                analyzer = explorer.create_image_analyzer(image_num)
                if analyzer:
                    analyzer.clear_roi()
                    print("✅ ROI cleared successfully")
                else:
                    print("Failed to clear ROI")
                    
            else:
                print("No ROI operation specified. Use --set, --analyze, --interactive, or --clear.")
                
        except Exception as e:
            print(f"Error with ROI operation: {e}")
    
    def _command_compare(self):
        """Compare images command."""
        filename = self.args.file
        image1 = self.args.image1
        image2 = self.args.image2
        method = self.args.method
        save_file = self.args.save
        show_image = self.args.show
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            print(f"🔄 Comparing images {image1} and {image2} using {method} method...")
            
            comparison = explorer.compare_images(image1, image2, method)
            
            if comparison is None:
                print("Image comparison failed.")
                return
                
            print(f"✅ Image comparison completed")
            print(f"Comparison statistics:")
            print(f"  Min difference: {comparison.min():.4f}")
            print(f"  Max difference: {comparison.max():.4f}")
            print(f"  Mean difference: {comparison.mean():.4f}")
            print(f"  Std difference: {comparison.std():.4f}")
            
            # Save if requested
            if save_file:
                print(f"💾 Saving comparison result to: {save_file}")
                plt.imsave(save_file, comparison, cmap='gray')
                print(f"✅ Image saved successfully")
                
            # Show if requested
            if show_image:
                print("🖼️  Displaying comparison result...")
                plt.figure(figsize=(10, 8))
                plt.imshow(comparison, cmap='gray')
                plt.title(f"Image Comparison ({method}) - Images {image1} vs {image2}")
                plt.colorbar()
                plt.show()
                
        except Exception as e:
            print(f"Error comparing images: {e}")
    
    def _command_histogram(self):
        """Histogram command."""
        filename = self.args.file
        image_num = self.args.image
        bins = self.args.bins
        log_scale = self.args.log
        save_file = self.args.save
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            print(f"📊 Creating histogram for image {image_num}...")
            
            histogram_data = explorer.get_image_histogram(image_num, bins)
            
            if histogram_data is None:
                print("Histogram creation failed.")
                return
                
            # Display histogram
            plt.figure(figsize=(12, 6))
            
            # Find the center of each bin
            bin_centers = [(histogram_data['bin_edges'][i] + histogram_data['bin_edges'][i+1]) / 2 
                          for i in range(len(histogram_data['bin_edges']) - 1)]
            
            if log_scale:
                plt.semilogy(bin_centers, histogram_data['hist'], 'b-')
                plt.ylabel('Log Count')
            else:
                plt.plot(bin_centers, histogram_data['hist'], 'b-')
                plt.ylabel('Count')
                
            plt.xlabel('Pixel Value')
            plt.title(f"Histogram - Image {image_num}")
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            stats = explorer.analyze_image(image_num)
            if stats:
                plt.figtext(0.7, 0.7, f"Mean: {stats['mean']:.2f}", bbox=dict(facecolor='white', alpha=0.8))
                plt.figtext(0.7, 0.65, f"Std: {stats['std']:.2f}", bbox=dict(facecolor='white', alpha=0.8))
                plt.figtext(0.7, 0.6, f"Min: {stats['min']:.2f}", bbox=dict(facecolor='white', alpha=0.8))
                plt.figtext(0.7, 0.55, f"Max: {stats['max']:.2f}", bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            # Save if requested
            if save_file:
                print(f"💾 Saving histogram data to: {save_file}")
                import json
                with open(save_file, 'w') as f:
                    json.dump(histogram_data, f, indent=2)
                print(f"✅ Histogram data saved successfully")
                
        except Exception as e:
            print(f"Error creating histogram: {e}")
    
    def _command_features(self):
        """Features command."""
        filename = self.args.file
        image_num = self.args.image
        method = self.args.method
        json_output = self.args.json
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            print(f"🔍 Detecting {method} features in image {image_num}...")
            
            features = explorer.detect_features(image_num, method)
            
            if features is None:
                print("Feature detection failed.")
                return
                
            if json_output:
                import json
                print(json.dumps(features, indent=2))
            else:
                print(f"✅ Feature detection completed")
                print(f"Found {features.get('count', 0)} features")
                
                if method == 'blob':
                    print(f"Blob features:")
                    for i, blob in enumerate(features.get('blobs', [])[:10]):  # Show first 10
                        print(f"  Blob {i+1}: x={blob['x']}, y={blob['y']}, sigma={blob['sigma']:.2f}")
                    if len(features.get('blobs', [])) > 10:
                        print(f"  ... and {len(features.get('blobs', [])) - 10} more")
                        
                elif method == 'corner':
                    print(f"Corner features:")
                    for i, corner in enumerate(features.get('corners', [])[:10]):  # Show first 10
                        print(f"  Corner {i+1}: x={corner['x']}, y={corner['y']}")
                    if len(features.get('corners', [])) > 10:
                        print(f"  ... and {len(features.get('corners', [])) - 10} more")
                        
                elif method == 'edge':
                    print(f"Edge features:")
                    print(f"  Edge density: {features.get('edge_density', 0):.4f}")
                    edge_pixels = features.get('edge_pixels', ([], []))
                    print(f"  Total edge pixels: {len(edge_pixels[0]) if edge_pixels else 0}")
                
        except Exception as e:
            print(f"Error detecting features: {e}")
    
    def _command_profile(self):
        """Profile command."""
        filename = self.args.file
        image_num = self.args.image
        direction = self.args.direction
        position = self.args.position
        save_file = self.args.save
        
        try:
            explorer = TDMSFileExplorer(filename)
            
            if not explorer.has_image_data():
                print("No image data found in file.")
                return
                
            print(f"📈 Getting {direction} profile for image {image_num}...")
            
            profile_data = explorer.get_image_profile(image_num, direction, position)
            
            if profile_data is None:
                print("Profile extraction failed.")
                return
                
            # Display profile
            plt.figure(figsize=(12, 6))
            plt.plot(profile_data['x'], profile_data['y'], 'b-')
            plt.xlabel('Position')
            plt.ylabel('Intensity')
            
            if position is not None:
                plt.title(f"{direction.capitalize()} Profile at position {position} - Image {image_num}")
            else:
                plt.title(f"{direction.capitalize()} Profile (average) - Image {image_num}")
                
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Save if requested
            if save_file:
                print(f"💾 Saving profile data to: {save_file}")
                import json
                with open(save_file, 'w') as f:
                    json.dump(profile_data, f, indent=2)
                print(f"✅ Profile data saved successfully")
                
        except Exception as e:
            print(f"Error getting profile: {e}")


def main():
    """Main entry point for CLI."""
    cli = TDMS_CLI()
    cli.parse_args()
    cli.run()


if __name__ == "__main__":
    main()