import struct
import matplotlib.pyplot as plt

# File path
file_path = './202110280850.dat'

# Read binary data from file
with open(file_path, 'rb') as file:
    file_data = file.read()

# Extract the text part (assumed to be up to the first 38 bytes)
text_part = file_data[:38].decode('utf-8', errors='ignore')

# Extract date and time from the text part
# Example format: "Stacja ELF ELA1H28.10.2021 08:501 3 T:"
date_time = text_part.split()[2][5:] + ' ' + text_part.split()[3][:5]  # Extract "28.10.2021" and "08:50"
formatted_datetime = date_time.replace('.', '-')  # Change date format to "28-10-2021"

print(text_part.split()[1])
print(text_part.split()[2])
print(text_part.split()[3])

# Print the text part
print("Text Part of the File:")
print(text_part)

# The text part ends at byte 38, so we'll extract the binary data from there
binary_data = file_data[38:]

# Unpack the binary data as 8-bit unsigned integers
data_format_8bit = 'B'  # 'B' for 8-bit unsigned integers
num_values_8bit = len(binary_data) // struct.calcsize(data_format_8bit)
unpacked_data_8bit = struct.unpack(f'{num_values_8bit}{data_format_8bit}', binary_data)
first_measurements = unpacked_data_8bit[:300]

# Plot the unpacked data
plt.figure(figsize=(20, 5))
plt.plot(first_measurements, label='ELF Signal')
plt.title('ELF Data '+formatted_datetime)
plt.xlabel('Sample Index')
plt.ylabel('Signal Amplitude (8-bit)')
plt.legend()
plt.grid(True)
plt.show()

exit(0)