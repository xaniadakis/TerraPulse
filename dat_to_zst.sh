# shellcheck disable=SC1115
# !/bin/bash

# Define colors
RED='\033[0;31m'
BLUE='\033[0;94m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Start timer
start_time=$(date +%s)

echo -e "${YELLOW}Removing old directories...${NC}"
rm -rf ./output
make -s clean all -C ./c

# Step 3: Run dat_to_text
echo -e "${YELLOW}Running dat_to_text...${NC}"
./c/build/dat_to_text
echo -e "${GREEN}dat_to_text execution complete.${NC}"

# Step 3: Run dat_to_text
echo -e "${YELLOW}Fitting lorentzians...${NC}"
matlab -nodisplay -nosplash -nodesktop -r "cd('/home/vag/PycharmProjects/TerraPulse/matlab'); main(7); exit;"
echo -e "${GREEN}fitting complete.${NC}"

# Step 4: Run signal_to_psd.py
echo -e "${YELLOW}Running signal_to_psd...${NC}"
python3 ./py/signal_to_psd.py
echo -e "${GREEN}signal_to_psd execution complete.${NC}"

# End timer
end_time=$(date +%s)
execution_time=$(($end_time - $start_time))
# Print total execution time
echo -e "${GREEN}Total execution time: ${execution_time} seconds${NC}"

# Step 5: Run plot_zst.py
echo -e "${YELLOW}Running plot_zst.py...${NC}"
python3 ./py/plot_zst.py
echo -e "${GREEN}plot_zst.py execution complete.${NC}"
