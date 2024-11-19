# shellcheck disable=SC1115
# !/bin/bash

# Define colors
RED='\033[0;31m'
BLUE='\033[0;94m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for mode and input directory
if [ "$#" -lt 3 ]; then
  echo -e "${RED}Usage: $0 <mode: hel|pol> <input_dir> <output_dir>${NC}"
  exit 1
fi

mode=$1
input_dir=$2
output_dir=$3

# Validate mode
if [[ "$mode" != "hel" && "$mode" != "pol" ]]; then
  echo -e "${RED}Invalid mode: $mode. Use 'hel' or 'pol'.${NC}"
  exit 1
fi

# Start timer
start_time=$(date +%s)

# Clean and prepare
echo -e "${YELLOW}Removing old directories...${NC}"
rm -rf "$output_dir"
make -s clean all -C ./c

# Run signal_to_text based on mode
echo -e "${YELLOW}Running signal_to_text...${NC}"
./c/build/signal_to_text "$mode" "$input_dir" "$output_dir"
echo -e "${GREEN}signal_to_text execution complete.${NC}"

# Run signal_to_psd.py with the correct file type
echo -e "${YELLOW}Running signal_to_psd...${NC}"
python3 ./py/signal_to_psd.py --file-type "$mode" -d "$output_dir"
echo -e "${GREEN}signal_to_psd execution complete.${NC}"

# # Step 5: Run plot_zst.py
# echo -e "${YELLOW}Running plot_zst.py...${NC}"
# python3 ./py/plot_zst.py
# echo -e "${GREEN}plot_zst.py execution complete.${NC}"

# End timer
end_time=$(date +%s)
execution_time=$(($end_time - $start_time))
# Print total

