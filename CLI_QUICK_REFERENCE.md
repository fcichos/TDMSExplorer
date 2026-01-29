# TDMS Explorer CLI - Quick Reference

## Basic Commands

```bash
# List all TDMS files
python3 tdms_cli.py list

# Get file information
python3 tdms_cli.py info "file.tdms"

# Show image
python3 tdms_cli.py show "file.tdms" --image 0

# Export images
python3 tdms_cli.py export "file.tdms" output_dir

# Create animation
python3 tdms_cli.py animate "file.tdms" animation.mp4

# Get statistics
python3 tdms_cli.py stats "file.tdms" --images

# Access raw data
python3 tdms_cli.py raw "file.tdms" --group "Image" --channel "Image"
```

## Common Options

### List Command
```bash
python3 tdms_cli.py list [--dir DIRECTORY] [--details]
```

### Info Command
```bash
python3 tdms_cli.py info FILE [--json]
```

### Show Command
```bash
python3 tdms_cli.py show FILE [--image NUM] [--cmap COLOMAP] [--no-show]
```

### Animate Command
```bash
python3 tdms_cli.py animate FILE OUTPUT [--fps FPS] [--start START] [--end END] [--cmap COLOMAP] [--no-display]
```

### Export Command
```bash
python3 tdms_cli.py export FILE OUTPUT_DIR [--start START] [--end END] [--single NUM] [--prefix PREFIX] [--format FORMAT] [--cmap COLOMAP] [--overwrite]
```

### Raw Command
```bash
python3 tdms_cli.py raw FILE --group GROUP --channel CHANNEL [--info] [--save FILE]
```

### Stats Command
```bash
python3 tdms_cli.py stats FILE [--images] [--channels]
```

## Quick Examples

### 1. Explore a new TDMS file
```bash
python3 tdms_cli.py info "new_experiment.tdms"
python3 tdms_cli.py stats "new_experiment.tdms" --images --channels
```

### 2. Quick preview
```bash
python3 tdms_cli.py show "experiment.tdms" --image 0
python3 tdms_cli.py show "experiment.tdms" --image 50
python3 tdms_cli.py show "experiment.tdms" --image 99
```

### 3. Export for analysis
```bash
python3 tdms_cli.py export "experiment.tdms" analysis_images --format tiff
```

### 4. Create presentation animation
```bash
python3 tdms_cli.py animate "experiment.tdms" presentation.mp4 --fps 15 --cmap viridis
```

### 5. Extract raw data for custom processing
```bash
python3 tdms_cli.py raw "experiment.tdms" --group "Image" --channel "Image" --save experiment_data.npy
```

## Common Colormaps

- `gray` (default) - Grayscale
- `viridis` - Perceptually uniform
- `plasma` - High contrast
- `inferno` - High contrast
- `magma` - High contrast
- `jet` - Rainbow (not perceptually uniform)
- `hot` - Black-red-yellow-white
- `cool` - Cyan-magenta

## File Patterns

- Use quotes around filenames with spaces
- Image numbers are 0-indexed
- Frame ranges are inclusive
- Output directories are created automatically

## Help

```bash
# General help
python3 tdms_cli.py --help

# Command-specific help
python3 tdms_cli.py [command] --help
```

## Tips

1. **Start small**: Test with `--end 5` before processing entire files
2. **Check sizes**: Use `list --details` to see file sizes first
3. **Use `--no-show`**: For testing display commands
4. **Quote paths**: Always quote file paths with spaces
5. **Check ranges**: Use `info` command to see available frame ranges