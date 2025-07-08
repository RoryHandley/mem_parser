# Memory Parser

A Python tool for analyzing system memory usage from archived diagnostic files. Extracts and visualizes memory statistics and top process data to help identify memory consumption patterns.

## Features

- **Automated Archive Processing**: Extracts memory files from .tar.gz archives
- **Memory Trend Analysis**: Parses data to show free memory over time
- **Top Process Tracking**: Identifies the top 5 memory-consuming processes across multiple snapshots
- **Data Visualization**: Generates graphs showing memory trends and process usage patterns
- **Configurable Analysis**: Uses config.ini for customizable parsing parameters

## Usage

1. Place your .tar.gz diagnostic archive in the script directory
2. Configure analysis parameters in `config.ini`
3. Run the script:
   ```bash
   python Mem_parser.py
   ```

## Configuration

Edit `config.ini` to customize:
- **Files to analyze**: file patterns
- **Memory attributes**: Which memory metrics to track
- **Process data**: Number of processes per timestamp and columns to analyze

## Output

The tool generates two visualizations:
- **Memory Timeline**: Shows free memory trends over time with smoothed data
- **Process Analysis**: Tracks top 5 memory-consuming processes across snapshots

## Requirements

- Python 3.x
- pandas
- matplotlib
- configparser

## Data Processing

- **Deduplication**: Removes duplicate process entries, keeping highest memory usage
- **Smoothing**: Applies rolling window average to reduce noise in memory graphs
- **Aggregation**: Combines process data across multiple time snapshots for trend analysis
