# !/bin/bash

# Start timer
start_time=$(date +%s)

echo "Removing old directories..."
rm -rf ./output
make clean all -C ./c

# Step 3: Run dat_to_text
echo "Running dat_to_text..."
./c/dat_to_text
echo "dat_to_text execution complete."

# Step 4: Run signal_to_psd.py
echo "Running signal_to_psd..."
python3 ./py/signal_to_psd.py
echo "signal_to_psd execution complete."
# End timer
end_time=$(date +%s)
execution_time=$(($end_time - $start_time))
# Print total execution time
echo "Total execution time: ${execution_time} seconds"

# Step 5: Run plot_zst.py
echo "Running plot_zst.py..."
python3 ./py/plot_zst.py
echo "plot_zst.py execution complete."




