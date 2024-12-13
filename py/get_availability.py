import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import struct
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import os
from math import sqrt
from geopy.distance import geodesic
import pandas as pd
from tqdm import tqdm
import mplcursors
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import tkinter as tk
from tkinter import Toplevel
import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches

from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QListWidget, QCheckBox, QFileDialog, QAbstractItemView, QComboBox
)

# Variables to track the active annotation
active_annotation = None
close_button = None

class FileSelectorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("File Selector")
        self.setGeometry(200, 200, 800, 500)

        self.base_dirs = []  # List to hold multiple base directories
        self.selected_mode = None
        self.selected_dirs = []
        self.mode_buttons = {}

        # Layout setup
        main_layout = QVBoxLayout()

        # Step 1: Select base directories
        base_dir_layout = QHBoxLayout()
        self.base_dir_label = QLabel("Base Directories: None Selected")
        base_dir_button = QPushButton("Select Base Directories")
        base_dir_button.clicked.connect(self.select_base_directories)
        clear_base_dir_button = QPushButton("Clear Base Directories")
        clear_base_dir_button.clicked.connect(self.clear_base_directories)
        base_dir_layout.addWidget(self.base_dir_label)
        base_dir_layout.addWidget(base_dir_button)
        base_dir_layout.addWidget(clear_base_dir_button)
        main_layout.addLayout(base_dir_layout)

        # Step 3: Select directories (embedded in the window)
        self.directories_label = QLabel("Available Directories: None")
        main_layout.addWidget(self.directories_label)

        self.select_all_checkbox = QCheckBox("Select All Directories")
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all)
        main_layout.addWidget(self.select_all_checkbox)

        self.directories_list = QListWidget()
        self.directories_list.setSelectionMode(QAbstractItemView.MultiSelection)
        main_layout.addWidget(self.directories_list)

        # Step 2: Select file mode
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("Mode: Not Selected")
        mode_dropdown = QComboBox()
        mode_dropdown.addItems(["DAT", "POL", "SRD", "HEL"])
        mode_dropdown.currentTextChanged.connect(self.select_mode)
        mode_layout.addWidget(self.mode_label)
        mode_layout.addWidget(mode_dropdown)
        main_layout.addLayout(mode_layout)

        # mode_layout = QHBoxLayout()
        # self.mode_label = QLabel("Mode: Not Selected")
        # mode_layout.addWidget(self.mode_label)
        # self.mode_buttons["dat"] = QPushButton("DAT")
        # self.mode_buttons["dat"].clicked.connect(lambda: self.select_mode("dat"))
        # mode_layout.addWidget(self.mode_buttons["dat"])
        # self.mode_buttons["pol"] = QPushButton("POL")
        # self.mode_buttons["pol"].clicked.connect(lambda: self.select_mode("pol"))
        # mode_layout.addWidget(self.mode_buttons["pol"])
        # self.mode_buttons["hel"] = QPushButton("HEL")
        # self.mode_buttons["hel"].clicked.connect(lambda: self.select_mode("hel"))
        # mode_layout.addWidget(self.mode_buttons["hel"])
        # self.mode_buttons["srd"] = QPushButton("SRD")
        # self.mode_buttons["srd"].clicked.connect(lambda: self.select_mode("srd"))
        # mode_layout.addWidget(self.mode_buttons["srd"])
        # main_layout.addLayout(mode_layout)

        # Step 2: Select region
        region_layout = QHBoxLayout()
        self.region_label = QLabel("Region: Not Selected")
        region_dropdown = QComboBox()
        region_dropdown.addItems(["South (Parnon)", "North (Kalpaki)"])
        region_dropdown.currentTextChanged.connect(self.select_region)
        region_layout.addWidget(self.region_label)
        region_layout.addWidget(region_dropdown)
        main_layout.addLayout(region_layout)
        # Set the default selection and trigger the label update
        region_dropdown.setCurrentIndex(0)  # Select the first item
        self.select_region(region_dropdown.currentText(), automatic=True)  # Update the label

        # Action button
        self.process_button = QPushButton("Process Files")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(lambda: (self.close(), self.process_files()))
        main_layout.addWidget(self.process_button)

        self.setLayout(main_layout)

    def select_base_directories(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setWindowTitle("Select Base Directories")
        dialog.setDirectory('/mnt/e/')  # Start file dialog from /mnt/e/
        if dialog.exec_():
            selected_dirs = dialog.selectedFiles()
            for selected_dir in selected_dirs:
                # Prevent adding duplicate base directories
                if selected_dir not in self.base_dirs:
                    self.base_dirs.append(selected_dir)
            self.base_dir_label.setText(f"Base Directories: {', '.join(self.base_dirs)}")
            self.load_directories()
        self.check_ready_to_process()

    def clear_base_directories(self):
        self.base_dirs = []
        self.base_dir_label.setText("Base Directories: None Selected")
        self.directories_list.clear()
        self.directories_label.setText("Available Directories: None")
        self.select_all_checkbox.setChecked(False)
        self.check_ready_to_process()

    def load_directories(self):
        if not self.base_dirs:
            return

        all_directories = []
        for base_dir in self.base_dirs:
            top_level_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            all_directories.extend(top_level_dirs)

        self.directories_list.clear()
        self.directories_list.addItems(all_directories)
        self.directories_label.setText(f"Available Directories: {len(all_directories)} found")
        self.select_all_checkbox.setChecked(False)

    def toggle_select_all(self):
        is_checked = self.select_all_checkbox.isChecked()
        for index in range(self.directories_list.count()):
            item = self.directories_list.item(index)
            item.setSelected(is_checked)
        self.check_ready_to_process()

    def select_mode(self, mode):
    #     self.selected_mode = mode
    #     self.mode_label.setText(f"Mode: {self.selected_mode.upper()}")

    #     # Update button colors to reflect selection
    #     for key, button in self.mode_buttons.items():
    #         if key == mode:
    #             button.setStyleSheet("background-color: lightblue;")
    #         else:
    #             button.setStyleSheet("")
    #     self.check_ready_to_process()
        self.selected_mode = mode.lower()
        self.mode_label.setText(f"Mode: {self.selected_mode.upper()}")
        self.check_ready_to_process()

    def select_region(self, region, automatic=False):
        self.region_label.setText(f"Region: {region}")
        if "parnon".lower() in region.lower() or "south".lower() in region.lower():
            self.selected_region = "parnon"
        elif "kalpaki".lower() in region.lower() or "north".lower() in region.lower():
            self.selected_region = "kalpaki"
        if not automatic:
            self.check_ready_to_process()

    def check_ready_to_process(self):
        selected_dirs = [self.directories_list.item(i).text() for i in range(self.directories_list.count())
                         if self.directories_list.item(i).isSelected()]
        if self.base_dirs and self.selected_mode and selected_dirs:
            self.process_button.setEnabled(True)
        else:
            self.process_button.setEnabled(False)

    def process_files(self):
        self.selected_dirs = [self.directories_list.item(i).text() for i in range(self.directories_list.count())
                            if self.directories_list.item(i).isSelected()]

        if not (self.base_dirs and self.selected_mode and self.selected_dirs):
            return

        file_extension = f".{self.selected_mode}"
        if self.selected_mode == "hel" or self.selected_mode == "srd":
            interval_minutes = 10 
        else:
            interval_minutes = 5 

        files = []
        for base_dir in self.base_dirs:
            if self.selected_mode == "hel":
                files.extend(self.find_files_hel(base_dir, self.selected_dirs, file_extension))
            elif self.selected_mode == "srd":
                files.extend(self.find_files_srd(base_dir, self.selected_dirs, "SRD"))
            else:
                files.extend(self.find_files(base_dir, self.selected_dirs, file_extension))

        if not files:
            print(f"No {self.selected_mode} files found.")
            return

        # Find continuous periods and their overall start and end times
        continuous_periods = self.find_continuous_periods(files, interval_minutes)
        total_start = datetime.strptime(files[0][0].split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M")
        total_end = datetime.strptime(files[-1][0].split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M") + timedelta(minutes=interval_minutes)

        # Plot timeline
        self.plot_timeline(continuous_periods, total_start, total_end, interval_minutes)

    def get_srd_info(fn):
        """
        Extracts metadata from an SRD file.

        Parameters:
        fn (str): The file name (path) of the SRD data file.

        Returns:
        tuple:
            - date (float): The timestamp in seconds since the epoch, corrected if necessary.
            - fs (float): Sampling frequency in Hz.
            - ch (int): Channel information (0 or 1).
            - vbat (float): Battery voltage in volts.
            - ok (int): Success flag (1 if successful, 0 if not).
        """
        ok = 0
        date = 0
        ch = 0
        DATALOGGERID = int("CAD0FFE51513FFDC", 16)

        # Check file size
        if os.path.getsize(fn) < (2 * 512):
            return date, ch, ok

        with open(fn, 'rb') as fp:
            # Read DATALOGGERID
            ID = struct.unpack('Q', fp.read(8))[0]
            if ID != DATALOGGERID:
                print(f'File "{fn}" is not a logger record!')
                return date, ch, ok

            # Read timestamp components
            S = struct.unpack('B', fp.read(1))[0]
            MN = struct.unpack('B', fp.read(1))[0]
            H = struct.unpack('B', fp.read(1))[0]
            DAY = struct.unpack('B', fp.read(1))[0]
            D = struct.unpack('B', fp.read(1))[0]
            M = struct.unpack('B', fp.read(1))[0]
            Y = struct.unpack('B', fp.read(1))[0] + 1970

            # Convert to datetime
            date = datetime(Y, M, D, H, MN, S)

            # Define correction dates and adjust date if necessary
            t0 = datetime(2016, 1, 1)
            t1 = datetime(2017, 8, 1)
            t2 = datetime(2018, 8, 1)

            if t0 < date < t1:
                tslop = 480 / 600  # seconds-offset per day
                days_diff = (date - t0).days
                dt_seconds = days_diff * tslop
                date -= timedelta(seconds=dt_seconds)

            # Set to timestamp
            date = date.timestamp()

            # Read ch
            fp.seek(19, os.SEEK_SET)
            ch = struct.unpack('B', fp.read(1))[0]

            # Successfully read info
            ok = 1

        return date, ch, ok

    def find_files_srd(self, base_dir, selected_dirs, file_extension):
        all_files = []
        total_dirs = 0
        last_timestamp = None  # Variable to keep track of the last timestamp

        # Count total directories within selected paths
        with tqdm(desc=f"Counting directories in {base_dir}", unit="dir") as pbar:
            for subdir in selected_dirs:
                subdir_path = os.path.join(base_dir, subdir)
                for _, dirs, _ in os.walk(subdir_path):
                    total_dirs += len(dirs)
                    pbar.update(1)

        # Search only in selected directories
        with tqdm(total=total_dirs, desc=f"Scanning files in {base_dir}", unit="files") as pbar:
            for subdir in selected_dirs:
                subdir_path = os.path.join(base_dir, subdir)
                for root, dirs, files in os.walk(subdir_path):
                    for i, file in enumerate(files):
                        if file.endswith(file_extension):  # Check if the file has the right extension
                            file_path = os.path.join(root, file)

                            # Check if this is the first file or if it's sequential
                            if last_timestamp and self.is_sequential(file, files[i-1]):
                                # If it's sequential, just add 10 minutes to the last timestamp
                                last_timestamp += timedelta(minutes=10)
                                timestamp = last_timestamp
                                # print(f"seq: {timestamp.strftime("%Y%m%d%H%M")}, last: {last_timestamp.strftime("%Y%m%d%H%M")}")
                            else:
                            # If it's not sequential, call get_srd_info for the timestamp
                                date, ch, ok = get_srd_info(file_path)
                                if ok:
                                    timestamp = datetime.fromtimestamp(date)
                                    last_timestamp = timestamp  # Update last_timestamp
                            # Format timestamp to YYYYMMDDHHMM
                            timestamp_str = timestamp.strftime("%Y%m%d%H%M")
                            # all_files.append(timestamp_str+".SRD")
                            all_files.append((timestamp_str + ".SRD", ch+1))
                        pbar.update(1)

        # Sort files by timestamp (date)
        # return sorted(all_files)
        return sorted(all_files, key=lambda x: x[0])

    def is_sequential(self, current_file, previous_file):
        """
        Check if the current file's filename is sequential to the previous file's filename.
        Assumes filenames are numeric and have the pattern like "00001.SRD", "00002.SRD", etc.
        
        Parameters:
        current_file (str): The current file's name (e.g., "00002.SRD").
        previous_file (str): The previous file's name (e.g., "00001.SRD").
        
        Returns:
        bool: True if the files are sequential, False otherwise.
        """
        try:
            # Extract numeric part of filenames
            current_index = int(current_file.split('.')[0])
            previous_index = int(previous_file.split('.')[0])

            # Check if the current file index is exactly +1 from the previous file
            return current_index == previous_index + 1
        except ValueError:
            return False  # In case of filename parsing error, assume they are not sequential

    def find_files_hel(self, base_dir, selected_dirs, file_extension):
        all_files = []
        total_dirs = 0

        # Count total directories within selected paths
        with tqdm(desc=f"Counting directories in {base_dir}", unit="dir") as pbar:
            for subdir in selected_dirs:
                subdir_path = os.path.join(base_dir, subdir)
                for _, dirs, _ in os.walk(subdir_path):
                    total_dirs += len(dirs)
                    pbar.update(1)

        # Search only in selected directories
        with tqdm(total=total_dirs, desc=f"Scanning directories in {base_dir}", unit="dir") as pbar:
            for subdir in selected_dirs:
                subdir_path = os.path.join(base_dir, subdir)
                for root, dirs, files in os.walk(subdir_path):
                    first_file_flag = True          
                    for file in files:
                        if first_file_flag:
                            data = np.loadtxt(os.path.join(root, file), delimiter='\t')
                            ch = data.ndim
                            first_file_flag = False  # Set flag to False after the first file
                        if file.endswith(file_extension):
                            try:
                                # Check if the filename (ignoring the last two seconds digits) is in the correct format
                                datetime.strptime(file.split('.')[0][:12], "%Y%m%d%H%M")
                                all_files.append((os.path.join(root, file), ch))
                            except ValueError:
                                pass  # Skip files that don't match the format
                    pbar.update(1)
        # return sorted(all_files)
        return sorted(all_files, key=lambda x: x[0])

    def find_files(self, base_dir, selected_dirs, file_extension):
        all_files = []
        total_dirs = 0

        # Count total directories within selected paths
        with tqdm(desc=f"Counting directories in {base_dir}", unit="dir") as pbar:
            for subdir in selected_dirs:
                subdir_path = os.path.join(base_dir, subdir)
                for _, dirs, _ in os.walk(subdir_path):
                    total_dirs += len(dirs)
                    pbar.update(1)

        # Search only in selected directories
        with tqdm(total=total_dirs, desc=f"Scanning directories in {base_dir}", unit="dir") as pbar:
            for subdir in selected_dirs:
                subdir_path = os.path.join(base_dir, subdir)
                for root, dirs, files in os.walk(subdir_path):
                    first_file_flag = True          
                    for file in files:
                        if first_file_flag:
                            data = np.loadtxt(os.path.join(root, file), delimiter='\t')
                            ch = data.ndim
                            first_file_flag = False  # Set flag to False after the first file
                        if file.endswith(file_extension) and len(file.split('.')[0]) == 12:  # Check if in YYYYMMDDHHMM format
                            all_files.append((os.path.join(root, file), ch))
                    pbar.update(1)
        return sorted(all_files)

    def find_continuous_periods(self, files, interval_minutes=5):
        continuous_periods = []
        current_period = []
        last_time = None

        with tqdm(total=len(files), desc="Grouping files", unit="file") as pbar:
            for file in files:
                try:
                    # Extract timestamp and validate its format
                    timestamp_str = file[0].split(os.sep)[-1].split('.')[0][:12]  # Extract the first 12 characters
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                except ValueError as e:
                    print(f"Skipping invalid file: {file[0]} ({e})")
                    pbar.update(1)
                    continue  # Skip files with invalid formats

                if last_time and (timestamp - last_time > timedelta(minutes=interval_minutes)):
                    if current_period:
                        continuous_periods.append((current_period, file[1]))
                    current_period = []
                current_period.append(file[0])
                last_time = timestamp
                pbar.update(1)

            if current_period:
                continuous_periods.append((current_period, file[1]))

        # Divide periods by years
        # periods_by_year = {}
        # for period in continuous_periods:
        #     year = datetime.strptime(period[0].split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M").year
        #     if year not in periods_by_year:
        #         periods_by_year[year] = []
        #     periods_by_year[year].append(period)

        # # Print the time difference for the first 10 continuous periods of each year
        # for year, periods in periods_by_year.items():
        #     print(f"Year: {year}, periods: {len(periods)}")
        #     for i in range(min(10, len(periods) - 1)):
        #         last_item = periods[i][-1]
        #         first_item_next = periods[i + 1][0]

        #         last_time = datetime.strptime(last_item.split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M")
        #         first_time_next = datetime.strptime(first_item_next.split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M")

        #         time_difference = first_time_next - last_time
        #         print(f"Period {i + 1} to {i + 2}: {time_difference}")

        return continuous_periods

    def prepare_dobrowolsky_df(self):
        print(f"Shall get earthquakes for {self.selected_region} location!")
        output_dir = "/mnt/c/Users/shumann/Documents/GaioPulse/earthquakes_db/output"
        file_path = os.path.join(output_dir, f"dobrowolsky_{self.selected_region}.csv")

        # Check if file does not exist
        if os.path.exists(file_path):
            print(f"The file '{file_path}' exists.")
            dobrowolsky_df = pd.read_csv(file_path)
            return dobrowolsky_df.drop(columns=['DOBROWOLSKY'])
        else:
            print(f"The file '{file_path}' does not exist. Will generate it.")

            # Boolean constant to decide the distance calculation method
            USE_HYPOTENUSE = True
            DOBROWOLSKY_TOLERANCE_FACTOR = 0.1

            # Load all CSVs into a single DataFrame
            csv_files = [
                os.path.join(output_dir, f) for f in os.listdir(output_dir)
                if f.endswith(".csv") and f[0].isdigit()
            ]
            all_data = []

            print("Loading CSV files...")
            for csv_file in tqdm(csv_files, desc="Loading Files"):
                df = pd.read_csv(csv_file)
                all_data.append(df)

            # Combine all CSVs into one DataFrame
            print("Combining all CSVs into a single DataFrame...")
            combined_df = pd.concat(all_data, ignore_index=True)

            # Simplify header names
            print("Simplifying header names...")
            simplified_headers = {
                col: col.split('(')[0].strip().replace('.', '')
                for col in combined_df.columns
            }
            combined_df.rename(columns=simplified_headers, inplace=True)

            # Ensure latitude, longitude, and depth columns are float
            combined_df['LAT'] = pd.to_numeric(combined_df['LAT'], errors='coerce')
            combined_df['LONG'] = pd.to_numeric(combined_df['LONG'], errors='coerce')
            combined_df['DEPTH'] = pd.to_numeric(combined_df['DEPTH'], errors='coerce')

            # Define Parnon location
            parnon_location = (37.2609, 22.5847)
            kalpaki_location = (39.9126, 20.5888)

            if self.selected_region == "parnon":
                coil_location = parnon_location
            elif self.selected_region == "kalpaki":
                coil_location = kalpaki_location

            # Calculate distance for each row
            def calculate_distance(row):
                event_location = (row['LAT'], row['LONG'])
                depth = row['DEPTH']  # Depth in kilometers
                if pd.notnull(event_location[0]) and pd.notnull(event_location[1]):
                    surface_distance = geodesic(coil_location, event_location).kilometers
                    if USE_HYPOTENUSE and pd.notnull(depth):
                        return sqrt(surface_distance**2 + depth**2)  # Hypotenuse distance
                    return surface_distance  # Surface distance
                return None

            print(f"Calculating {'hypotenuse' if USE_HYPOTENUSE else 'surface'} distances to Parnon...")
            combined_df['COIL_DISTANCE'] = list(
                tqdm(combined_df.apply(calculate_distance, axis=1), desc="Calculating Distances", total=len(combined_df))
            )

            # Compute the preparation radius for each row
            combined_df['PREPARATION_RADIUS'] = 10**(0.43 * combined_df['MAGNITUDE'])

            # Round distances and preparation radius to 2 decimal places
            combined_df['COIL_DISTANCE'] = combined_df['COIL_DISTANCE'].round(2)
            combined_df['PREPARATION_RADIUS'] = combined_df['PREPARATION_RADIUS'].round(2)

            # Apply the Dobrowolsky law with tolerance
            def apply_dobrowolsky_law(row):
                # Add tolerance to the preparation radius
                tolerance = row['PREPARATION_RADIUS'] * DOBROWOLSKY_TOLERANCE_FACTOR
                effective_radius = row['PREPARATION_RADIUS'] + tolerance
                # Check if the distance is less than or equal to the preparation radius with tolerance
                if row['COIL_DISTANCE'] <= effective_radius:
                    return 1
                return 0

            print("Applying the Dobrowolsky law...")
            combined_df['DOBROWOLSKY'] = list(
                tqdm(combined_df.apply(apply_dobrowolsky_law, axis=1), desc="Applying Dobrowolsky", total=len(combined_df))
            )

            # Filter rows where Dobrowolsky law applies
            print("Filtering rows where Dobrowolsky law applies...")
            dobrowolsky_df = combined_df[combined_df['DOBROWOLSKY'] == 1]


            dobrowolsky_df["DATE"] = pd.to_datetime(dobrowolsky_df["DATE"])
            # Parse TIME as a timedelta (hours, minutes, seconds)
            dobrowolsky_df["TIME"] = pd.to_timedelta(dobrowolsky_df["TIME"].str.replace(' ', ':'))
            # Add TIME as timedelta to DATE to create DATETIME
            dobrowolsky_df["DATETIME"] = dobrowolsky_df["DATE"] + dobrowolsky_df["TIME"]
            # Drop the DATE, TIME columns
            dobrowolsky_df.drop(columns=["DATE", "TIME"], inplace=True)
            # Move the DATETIME column to the second position
            cols = list(dobrowolsky_df.columns)
            cols.insert(1, cols.pop(cols.index("DATETIME")))  # Move DATETIME to the second position
            dobrowolsky_df = dobrowolsky_df[cols]

            # Sort by date ascending
            print("Sorting Dobrowolsky-valid rows by date...")
            dobrowolsky_df = dobrowolsky_df.sort_values(by='DATETIME', ascending=True)

            # Add a unique ID column to combined DataFrame
            print("Adding unique ID column...")
            dobrowolsky_df.insert(0, 'ID', range(1, len(dobrowolsky_df) + 1))

            # Save the filtered DataFrame to a new CSV file
            dobrowolsky_csv = os.path.join(output_dir, f"dobrowolsky_{self.selected_region}.csv")
            dobrowolsky_df.to_csv(dobrowolsky_csv, index=False)
            print(f"Dobrowolsky valid rows saved to {dobrowolsky_csv}")

            return dobrowolsky_df.drop(columns=['DOBROWOLSKY'])

    def calculate_normalized_impact(self, M, CD, pr, 
                                    min_magnitude, max_magnitude, 
                                    min_coil_distance, max_coil_distance, 
                                    p=2):
        """
        Calculate the normalized impact of an event scaled between 0 and 1.

        Args:
        - M (float): Magnitude of the event.
        - CD (float): Distance of the event from you.
        - pr (float): Preparation radius.
        - min_magnitude (float): Minimum magnitude in the dataset.
        - max_magnitude (float): Maximum magnitude in the dataset.
        - min_coil_distance (float): Minimum distance in the dataset.
        - max_coil_distance (float): Maximum distance in the dataset.
        - p (float): Distance decay factor (default is 2).

        Returns:
        - float: Normalized impact value (0 if CD > pr).
        """
        if CD > pr:
            return 0

        # Normalize magnitude and distance
        norm_M = (M - min_magnitude) / (max_magnitude - min_magnitude)
        norm_CD = (CD - min_coil_distance) / (max_coil_distance - min_coil_distance)
        
        # Calculate normalized impact
        impact = norm_M / (norm_CD ** p) if norm_CD > 0 else norm_M  # Avoid division by zero
        
        # Ensure the output is scaled between 0 and 20
        return min(max(impact, 0), 20)

    def scale_linewidth(self, impact, min_impact=0.0, max_impact=20.0, min_width=0.25, max_width=2.5):
        # Prevent division by zero
        if max_impact - min_impact == 0:
            return max_width
        # Scale impact to the width range
        normalized = (impact - min_impact) / (max_impact - min_impact)
        return min_width + normalized * (max_width - min_width)

    def plot_timeline(self, continuous_periods, total_start, total_end, interval_minutes=5):

        dobrowolsky_df = self.prepare_dobrowolsky_df()

        # dobrowolsky_df = pd.read_csv('/mnt/c/Users/shumann/Documents/GaioPulse/earthquakes_db/output/dobrowolsky_valid_rows.csv')

        earthquakes_by_year = None
        if dobrowolsky_df is not None:
            # Convert DATE column to datetime
            dobrowolsky_df['DATETIME'] = pd.to_datetime(dobrowolsky_df['DATETIME'], errors='coerce')

            # Ensure no invalid dates are present
            if dobrowolsky_df['DATETIME'].isnull().any():
                print("Warning: Some rows have invalid dates and will be excluded.")
                dobrowolsky_df = dobrowolsky_df.dropna(subset=['DATETIME'])

            dobrowolsky_df['YEAR'] = dobrowolsky_df['DATETIME'].dt.year
            dobrowolsky_df['DAY_OF_YEAR'] = dobrowolsky_df['DATETIME'].dt.dayofyear

            # Group data by year
            earthquakes_by_year = dobrowolsky_df.groupby('YEAR')

        print(f"Start: {total_start}, End: {total_end}")

        # Assuming self.selected_mode contains the file type (e.g., 'dat', 'pol', 'hel')
        file_type = self.selected_mode.upper()  # Capitalize the file type for display
        
        # Group periods by year
        periods_by_year = {}
        total_days_with_data = set()  # Track total days across all years
        
        for period in continuous_periods:
            try:
                start_year = datetime.strptime(period[0][0].split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M").year
                if start_year not in periods_by_year:
                    periods_by_year[start_year] = []
                periods_by_year[start_year].append((period[0], period[1]))
            except ValueError as e:
                print(f"Skipping period due to invalid format: {period} ({e})")
                continue

        sorted_years = sorted(periods_by_year.keys())
        num_years = len(sorted_years)

        # Dynamically adjust the figure height for compactness but ensure it's large enough for visibility
        fig_height = min(2 * num_years, 12)  # Cap the total height to a reasonable desktop size
        fig, axes = plt.subplots(num_years, 1, figsize=(15, fig_height))  # No sharex=True

        # If there is only one year, make sure axes is still iterable
        if num_years == 1:
            axes = [axes]

        handled_labels = set()
        line_na, line_dc, line_sc, line_eq = None, None, None, None
        quake_annotations = {}

        # Iterate over each year and its periods
        for i, year in enumerate(sorted_years):
            ax = axes[i]
            year_periods = periods_by_year[year]

            # Determine start and end of the year
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31, 23, 59)

            last_end = year_start
            days_with_data = set()

            with tqdm(total=len(year_periods), desc=f"Plotting timeline for {year}", unit="period") as pbar:
                print(f"Year {year} has {len(year_periods)} periods")
                for period in year_periods:
                    try:
                        # Extract and validate the start timestamp
                        start_str = period[0][0].split(os.sep)[-1].split('.')[0][:12]
                        start = datetime.strptime(start_str, "%Y%m%d%H%M")

                        # Extract and validate the end timestamp
                        end_str = period[0][-1].split(os.sep)[-1].split('.')[0][:12]
                        end = datetime.strptime(end_str, "%Y%m%d%H%M") + timedelta(minutes=interval_minutes)

                        # Add days with data to the set
                        current_day = start
                        while current_day <= end:
                            days_with_data.add(current_day.date())
                            current_day += timedelta(days=1)

                         # Plot gap (unavailable period) in red
                        if start > last_end:
                            line_na, = ax.plot([last_end, start], [0.5, 0.5], color='red', lw=6, solid_capstyle='butt', label="NA")
                            if "NA" not in handled_labels:
                                handled_labels.add("NA")

                        # Plot continuous period
                        if period[1] == 1:
                            line_sc, = ax.plot([start, end], [0.5, 0.5], color='blue', lw=6, solid_capstyle='butt', label="Single Channel")
                            if "Single Channel" not in handled_labels:
                                handled_labels.add("Single Channel")
                        elif period[1] == 2:
                            line_dc, = ax.plot([start, end], [0.5, 0.5], color='green', lw=6, solid_capstyle='butt', label="Dual Channel")
                            if "Dual Channel" not in handled_labels:
                                handled_labels.add("Dual Channel")
                        else:
                            print(f"Invalid number of channels: {period[1]} found!")
                        last_end = end
                    except ValueError as e:
                        print(f"Skipping invalid file in period: {period} ({e})")
                        continue
                    pbar.update(1)

                # Plot any final gap after the last period
                if last_end < year_end:
                    ax.plot([last_end, year_end], [0.5, 0.5], color='red', lw=6, solid_capstyle='butt')

            # Plot earthquake occurrences for the year as arrows
            # if earthquakes_by_year is not None:
            #     if year in earthquakes_by_year.groups:
            #         earthquake_days = dobrowolsky_df[dobrowolsky_df['YEAR'] == year]['DATE']
            #         for quake_date in earthquake_days:
            #             ax.annotate(
            #                 '',  # No text, just an arrow
            #                 xy=(quake_date, 0.5),  # End of the arrow
            #                 xytext=(quake_date, 1.0),  # Start of the arrow
            #                 arrowprops=dict(facecolor='brown', arrowstyle='-|>', lw=0.7, color='brown'),
            #             )
            #         if "Earthquake" not in handled_labels:
            #             handled_labels.add("Earthquake")
            #             line_eq, = ax.plot([], [], color='orange', label='Earthquake')  # For legend
            # Plot earthquake occurrences for the year as arrows
            # Plot earthquake occurrences for the year as arrows
            min_magnitude = dobrowolsky_df['MAGNITUDE'].min()
            max_magnitude = dobrowolsky_df['MAGNITUDE'].max()

            min_coil_distance = dobrowolsky_df['COIL_DISTANCE'].min()
            max_coil_distance = dobrowolsky_df['COIL_DISTANCE'].max()

            if earthquakes_by_year is not None:
                if year in earthquakes_by_year.groups:
                    earthquakes_in_year = dobrowolsky_df[dobrowolsky_df['YEAR'] == year]
                    for _, row in earthquakes_in_year.iterrows():
                        quake_date = row['DATETIME']
                        magnitude = row['MAGNITUDE']
                        quake_id = row['ID']  # Assuming the DataFrame has an 'ID' column
                        coil_distance = row['COIL_DISTANCE']
                        preparation_radius = row['PREPARATION_RADIUS']
                        depth = row['DEPTH']
                        
                        impact = self.calculate_normalized_impact(magnitude, coil_distance, preparation_radius, 
                                                                  min_magnitude, max_magnitude, 
                                                                  min_coil_distance, max_coil_distance)
                        # Example usage:
                        arrow_linewidth = self.scale_linewidth(impact=impact)

                        print(f" quake: {quake_id} has width of: {arrow_linewidth} with impact={impact}")
                        # Plot the earthquake arrow
                        arrow = ax.annotate(
                            '',  # No text, just an arrow
                            xy=(quake_date, 0.5),  # End of the arrow
                            xytext=(quake_date, 0.8),  # Start of the arrow
                            arrowprops=dict(facecolor='brown', arrowstyle='-|>', lw=arrow_linewidth, color='brown'),
                        )
                        
                        # Plot the earthquake ID as a text label
                        ax.text(
                            quake_date, 0.8,  # Position the text slightly above the arrow
                            str(quake_id),  # Convert ID to string
                            fontsize=10,  # Adjust font size as needed
                            ha='center',  # Center the text horizontally
                            color='black',  # Set the text color
                        )

                        # Extract human-readable date
                        human_readable_date = quake_date.strftime("%A, %d %B %Y")  # E.g., 'Monday, 15 February 2016'

                        # Extract 24-hour time
                        time_24_hour = quake_date.strftime("%H:%M:%S")  # E.g., '18:55:00'

                        # Store annotation and its data in the dictionary
                        quake_annotations[(mdates.date2num(quake_date), 0.65)] = {
                            "ID": quake_id,
                            "Date": human_readable_date,
                            "Time": time_24_hour,
                            "Magnitude": magnitude,
                            "CoilDistance": coil_distance,
                            "PreparationRadius": preparation_radius,
                            "Depth": depth,
                        }
                    
                    # if "Earthquake" not in handled_labels:
                    #     handled_labels.add("Earthquake")
                    #     line_eq, = ax.plot([], [], color='brown', label='Earthquake')  # For legend

                    from matplotlib.lines import Line2D

                    if "Earthquake" not in handled_labels:
                        handled_labels.add("Earthquake")
                        # Use Line2D with an arrow marker for the legend
                        line_eq = Line2D(
                            [], [], 
                            color='brown', 
                            marker=r'$\rightarrow$',  # Arrow marker
                            markersize=15,  # Adjust size as needed
                            linestyle='None',  # No connecting line
                            label='Earthquake'
                        )
                        # ax.legend(handles=[arrow_marker], loc='lower center')  # Add to legend


            # Attach mplcursors for hover interactivity
            # cursor = mplcursors.cursor(hover=True)

            # @cursor.connect("add")
            # def on_hover(sel):
            #     for annotation, info_text in quake_annotations:
            #         if sel.artist == annotation.arrow_patch:  # Match the arrow object
            #             sel.annotation.set_text(info_text)
            #             sel.annotation.get_bbox_patch().set(alpha=0.8, facecolor='lightyellow')
            #             break

            # Set x-axis limits and ticks for each subplot
            ax.set_xlim(year_start, year_end)

            # Set two ticks per month: first day and a date in the second week
            weekly_ticks = []
            for month in range(1, 13):
                month_start = datetime(year, month, 1)
                month_end = datetime(year, month, 1) + relativedelta(months=1) - timedelta(days=1)
                weekly_ticks.extend(pd.date_range(month_start, month_end, freq='W-MON'))

            ax.set_xticks(weekly_ticks)

            # Custom formatter: Show the week number of the year
            def custom_date_formatter(x, pos):
                date = mdates.num2date(x)  # Convert x-axis ticks to datetime
                if date.day <= 7:  # First tick of the month
                    return date.strftime('%b')
                else:  # Second tick of the month
                    return f"{date.strftime('%U')}"

            ax.xaxis.set_major_formatter(FuncFormatter(custom_date_formatter))

            # Ensure y-axis is visible enough to display data
            ax.set_yticks([])
            ax.set_ylim(0, 1)

            # Display the number of days with data in the title
            days_count = len(days_with_data)
            for tick in ax.get_xticklabels():
                if any(month in tick.get_text() for month in [
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']):
                    tick.set_rotation(60)  # Rotate month names more
                else:
                    tick.set_rotation(0)

            ax.set_title(f"Timeline for {year} ({days_count} days available)", fontsize=12, pad=5)
            ax.grid(visible=True, axis='x', linestyle='--', alpha=0.5)
            # Add the days from this year to the total days count
            total_days_with_data.update(days_with_data)

        # Display total number of available days across all years at the bottom of the figure
        total_days = len(total_days_with_data)
        
        # Set the figure's title to show the total number of days and the file type
        if file_type == "HEL":
            plt.suptitle(f"Hellenic Logger Processed Data (available in total: {total_days} days)", fontsize=14, fontweight='bold', y=0.98)
        elif file_type == "SRD":
            plt.suptitle(f"Hellenic Logger Original Data (available in total: {total_days} days)", fontsize=14, fontweight='bold', y=0.98)
        elif file_type == "POL":
            plt.suptitle(f"Polski Logger Processed Data (available in total: {total_days} days)", fontsize=14, fontweight='bold', y=0.98)
        elif file_type == "DAT":
            plt.suptitle(f"Polski Logger Original Data (available in total: {total_days} days)", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(pad=3.0)  # Adjust padding to ensure the title doesn't overlap

        # Adjust labels and layout after plotting all years
        # Add the legend to the figure level

        legend_lines = []
        legend_labels = []
        
        if line_na:
            legend_lines.append(line_na)
            legend_labels.append("N/A")
        if line_dc:
            legend_lines.append(line_dc)
            legend_labels.append("Dual Channel")
        if line_sc:
            legend_lines.append(line_sc)
            legend_labels.append("Single Channel")
        if line_eq:
            legend_lines.append(line_eq)
            legend_labels.append("Earthquake")

        # Function to check if a click is within a radius of an arrow
        def is_within_radius(click_x, click_y, arrow_x, arrow_y, radius):
            return np.sqrt((click_x - arrow_x) ** 2 + (click_y - arrow_y) ** 2) <= radius

        # Function to handle click events
        def on_click(event):
            click_x = event.xdata
            click_y = event.ydata
            if click_x is None or click_y is None:  # Ignore clicks outside the axes
                return

            radius = 10  # Set the radius for detection
            for (arrow_x, arrow_y), quake_data in quake_annotations.items():
                if is_within_radius(click_x, click_y, arrow_x, arrow_y, radius):
                    # Remove any existing annotation box
                    for child in ax.get_children():
                        if isinstance(child, AnchoredText):
                            child.remove()

                    # Create a new annotation box
                    details_text = (
                        f"Event ID: {quake_data['ID']}\n"
                        f"Date: {quake_data['Date']}\n"
                        f"Time: {quake_data['Time']}\n"
                        f"Magnitude: {quake_data['Magnitude']}\n"
                        f"Depth: {quake_data['Depth']}\n"
                        f"Distance from {self.selected_region.capitalize()} coil: {quake_data['CoilDistance']}\n"
                        f"Preparation Radius: {quake_data['PreparationRadius']}\n"
                    )
                    annotation_box = AnchoredText(
                        details_text,
                        loc='upper left',  # Adjust location as needed
                        prop=dict(size=10),
                        frameon=True,
                        pad=0.5,
                        borderpad=0.5,
                    )
                    annotation_box.patch.set_boxstyle("round,pad=0.3")
                    annotation_box.patch.set_alpha(0.85)
                    annotation_box.patch.set_facecolor("lightyellow")

                    # Add the annotation box to the axes
                    ax.add_artist(annotation_box)
                    plt.draw()  # Redraw the plot to show the annotation
                    break

        # Connect the click and pick event handlers
        fig.canvas.mpl_connect("button_press_event", on_click)

        fig.legend(legend_lines, legend_labels, loc="lower center", ncol=3, fontsize=10)

        # Adjust labels and layout after plotting all years
        plt.xlabel("Months", fontsize=10, color="white")  # Set the color of the x-axis label
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = QApplication([])
    window = FileSelectorApp()
    window.show()
    app.exec_()
