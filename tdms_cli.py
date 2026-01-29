#!/usr/bin/env python3
"""
TDMS Explorer CLI - Command Line Interface

A comprehensive command-line tool for working with TDMS files.
Provides functionality to explore, visualize, and export data from TDMS files.
"""

import sys
import os
import argparse
import textwrap
from typing import List, Optional

# Add current directory to Python path to import the module
sys.path.insert(0, '.')

try:
    from tdms_explorer import TDMSFileExplorer, list_tdms_files, create_animation_from_tdms
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
  python tdms_cli.py list
  
  # Show information about a specific file
  python tdms_cli.py info "file.tdms"
  
  # Display first image from a file
  python tdms_cli.py show "file.tdms" --image 0
  
  # Export all images to directory
  python tdms_cli.py export "file.tdms" output_dir
  
  # Create animation
  python tdms_cli.py animate "file.tdms" animation.mp4 --fps 10
  
  # Get raw data info
  python tdms_cli.py raw "file.tdms" --group "Image" --channel "Image"
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
        show_parser.add_argument('--no-show', action='store_true', help='Don\'t display image (just test)')
        
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
            
            print(f"Displaying image {image_num} from {filename}")
            print(f"Image shape: {images.shape[1]}x{images.shape[2]}")
            print(f"Data range: {images[image_num].min():.2f} to {images[image_num].max():.2f}")
            
            if not no_show:
                explorer.display_image(image_num, cmap=cmap)
            else:
                print("Image display skipped (--no-show flag)")
                
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


def main():
    """Main entry point for CLI."""
    cli = TDMS_CLI()
    cli.parse_args()
    cli.run()


if __name__ == "__main__":
    main()