# TDMS Explorer CLI - Quick Reference

## Basic Commands

```bash
# List all TDMS files
tdms-explorer list

# Get file information
tdms-explorer info "file.tdms"

# Show image
tdms-explorer show "file.tdms" --image 0

# Export images
tdms-explorer export "file.tdms" output_dir

# Create animation
tdms-explorer animate "file.tdms" animation.mp4

# Get statistics
tdms-explorer stats "file.tdms" --images

# Access raw data
tdms-explorer raw "file.tdms" --group "Image" --channel "Image"
```

## Common Options

### List Command
```bash
tdms-explorer list [--dir DIRECTORY] [--details]
```

### Info Command
```bash
tdms-explorer info FILE [--json]
```

### Show Command
```bash
tdms-explorer show FILE [--image NUM] [--cmap COLOMAP] [--no-show] [--save FILE]
```

### Animate Command
```bash
tdms-explorer animate FILE OUTPUT [--fps FPS] [--start START] [--end END] [--cmap COLOMAP] [--no-display]
```

### Export Command
```bash
tdms-explorer export INPUT [OUTPUT]
  [--start START] [--end END] [--single NUM]
  [--prefix PREFIX] [--base-name NAME]
  [--format FORMAT] [--cmap COLOMAP]
  [--dtype {uint8,uint16,float32}] [--normed | --no-normed]
  [--workers N] [--to-mp4] [--fps FPS]
  [-f | --overwrite]
  [--list-structure]
  [--start-index N] [--num-files N]
```

Notes:
- Default single-file names: `output_000.png`, `output_001.png`, …
- `--prefix frame_` → `frame_000.png` (0-based, inclusive `--end`)
- `--base-name exp` → `exp_001.png` (1-based stem style)
- `--dtype float32 --format png` saves uint16 PNG with a note; use `--format tiff` for float32
- `convert` is a deprecated alias for `export` (use `-o` for output dir)

### Raw Command
```bash
tdms-explorer raw FILE --group GROUP --channel CHANNEL [--info] [--save FILE]
```

### Stats Command
```bash
tdms-explorer stats FILE [--images] [--channels]
```

## Quick Examples

### 1. Explore a new TDMS file
```bash
tdms-explorer info "new_experiment.tdms"
tdms-explorer stats "new_experiment.tdms" --images --channels
```

### 2. Quick preview
```bash
tdms-explorer show "experiment.tdms" --image 0
tdms-explorer show "experiment.tdms" --image 50
tdms-explorer show "experiment.tdms" --image 99
```

### 3. Export for analysis
```bash
tdms-explorer export "experiment.tdms" analysis_images --format tiff --dtype float32
tdms-explorer export "experiment.tdms" analysis_images --start 0 --end 10
```

### 4. Batch export + video
```bash
tdms-explorer export "run_{:03d}.tdms" output/ --start-index 1 --num-files 5 --to-mp4 --fps 30
```

### 5. Create presentation animation
```bash
tdms-explorer animate "experiment.tdms" presentation.mp4 --fps 15 --cmap viridis
```

### 6. Extract raw data for custom processing
```bash
tdms-explorer raw "experiment.tdms" --group "Image" --channel "Image" --save experiment_data.npy
```

## Common Colormaps

- `gray` - Grayscale
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
- `--end` is inclusive for `--prefix` exports
- Output directories are created automatically

## Help

```bash
tdms-explorer --help
tdms-explorer export --help
```

## Tips

1. Start small: test with `--end 5` before processing entire files
2. Check sizes: use `list --details` to see file sizes first
3. Use `--save` on `show` when display is unavailable
4. Quote paths with spaces
5. Prefer `export` over deprecated `convert`
