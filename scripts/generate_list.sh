#!/usr/bin/env bash
#
# Author: Daniel Kurniadi
# Description: generate a text file listing all videos dataset in a folder
#
# Usage: $ bash generate_list.sh <SRC_VIDEO_FOLDER> <DEST_FILENAME>
# e.g.	 $ bash generate_list.sh ../Data/AGG01/PositiveSamples/ ./data/raw/agg01_positive.txt
#
SRC=$1
DST=$2

# lists all video extensions here
videxts_regx="\.avi$|\.webm$|\.f4v$|\.flv$|\.mp4$"

# lists all videos in source folder
find $SRC -type f | grep -E $videxts_regx >> $DST 

