import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import struct


from PyQt5.QtWidgets import (
    QApplication, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QListWidget, QCheckBox, QFileDialog, QAbstractItemView
)

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
    DATALOGGERID = int("CAD0FFE51513FFDC", 16)

    # Check file size
    if os.path.getsize(fn) < (2 * 512):
        return date, ok

    with open(fn, 'rb') as fp:
        # Read DATALOGGERID
        ID = struct.unpack('Q', fp.read(8))[0]
        if ID != DATALOGGERID:
            print(f'File "{fn}" is not a logger record!')
            return date, ok

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

        # Successfully read info
        ok = 1

    return date, ok



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
        mode_layout.addWidget(self.mode_label)

        self.mode_buttons["dat"] = QPushButton("DAT")
        self.mode_buttons["dat"].clicked.connect(lambda: self.select_mode("dat"))
        mode_layout.addWidget(self.mode_buttons["dat"])

        self.mode_buttons["pol"] = QPushButton("POL")
        self.mode_buttons["pol"].clicked.connect(lambda: self.select_mode("pol"))
        mode_layout.addWidget(self.mode_buttons["pol"])

        self.mode_buttons["hel"] = QPushButton("HEL")
        self.mode_buttons["hel"].clicked.connect(lambda: self.select_mode("hel"))
        mode_layout.addWidget(self.mode_buttons["hel"])

        self.mode_buttons["srd"] = QPushButton("SRD")
        self.mode_buttons["srd"].clicked.connect(lambda: self.select_mode("srd"))
        mode_layout.addWidget(self.mode_buttons["srd"])

        main_layout.addLayout(mode_layout)

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
        self.selected_mode = mode
        self.mode_label.setText(f"Mode: {self.selected_mode.upper()}")

        # Update button colors to reflect selection
        for key, button in self.mode_buttons.items():
            if key == mode:
                button.setStyleSheet("background-color: lightblue;")
            else:
                button.setStyleSheet("")
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
        interval_minutes = 10 if self.selected_mode == "hel" else 5  # Adjust interval based on mode

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
        total_start = datetime.strptime(files[0].split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M")
        total_end = datetime.strptime(files[-1].split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M") + timedelta(minutes=interval_minutes)

        # Plot timeline
        self.plot_timeline(continuous_periods, total_start, total_end, interval_minutes)

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
        with tqdm(total=total_dirs, desc=f"Scanning directories in {base_dir}", unit="dir") as pbar:
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
                                date, ok = get_srd_info(file_path)
                                if ok:
                                    timestamp = datetime.fromtimestamp(date)
                                    last_timestamp = timestamp  # Update last_timestamp
                            # Format timestamp to YYYYMMDDHHMM
                            timestamp_str = timestamp.strftime("%Y%m%d%H%M")
                            all_files.append(timestamp_str+".SRD")
                        pbar.update(1)

        # Sort files by timestamp (date)
        return sorted(all_files)

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
                    for file in files:
                        if file.endswith(file_extension):
                            try:
                                # Check if the filename (ignoring the last two seconds digits) is in the correct format
                                datetime.strptime(file.split('.')[0][:12], "%Y%m%d%H%M")
                                all_files.append(os.path.join(root, file))
                            except ValueError:
                                pass  # Skip files that don't match the format
                    pbar.update(1)
        return sorted(all_files)


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
                    for file in files:
                        if file.endswith(file_extension) and len(file.split('.')[0]) == 12:  # Check if in YYYYMMDDHHMM format
                            all_files.append(os.path.join(root, file))
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
                    timestamp_str = file.split(os.sep)[-1].split('.')[0][:12]  # Extract the first 12 characters
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")
                except ValueError as e:
                    print(f"Skipping invalid file: {file} ({e})")
                    pbar.update(1)
                    continue  # Skip files with invalid formats

                if last_time and (timestamp - last_time != timedelta(minutes=interval_minutes)):
                    if current_period:
                        continuous_periods.append(current_period)
                    current_period = []
                current_period.append(file)
                last_time = timestamp
                pbar.update(1)

            if current_period:
                continuous_periods.append(current_period)
        # print(f"cont periods: {len(continuous_periods)}")
        return continuous_periods


    def plot_timeline(self, continuous_periods, total_start, total_end, interval_minutes=5):
        print(f"Start: {total_start}, End: {total_end}")

        # Assuming self.selected_mode contains the file type (e.g., 'dat', 'pol', 'hel')
        file_type = self.selected_mode.upper()  # Capitalize the file type for display
        
        # Group periods by year
        periods_by_year = {}
        total_days_with_data = set()  # Track total days across all years
        
        for period in continuous_periods:
            try:
                start_year = datetime.strptime(period[0].split(os.sep)[-1].split('.')[0][:12], "%Y%m%d%H%M").year
                if start_year not in periods_by_year:
                    periods_by_year[start_year] = []
                periods_by_year[start_year].append(period)
            except ValueError as e:
                print(f"Skipping period due to invalid format: {period} ({e})")
                continue

        sorted_years = sorted(periods_by_year.keys())
        num_years = len(sorted_years)

        # Dynamically adjust the figure height for compactness but ensure it's large enough for visibility
        fig_height = min(1.5 * num_years, 12)  # Cap the total height to a reasonable desktop size
        fig, axes = plt.subplots(num_years, 1, figsize=(15, fig_height))  # No sharex=True

        # If there is only one year, make sure axes is still iterable
        if num_years == 1:
            axes = [axes]

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
                for period in year_periods:
                    try:
                        # Extract and validate the start timestamp
                        start_str = period[0].split(os.sep)[-1].split('.')[0][:12]
                        start = datetime.strptime(start_str, "%Y%m%d%H%M")

                        # Extract and validate the end timestamp
                        end_str = period[-1].split(os.sep)[-1].split('.')[0][:12]
                        end = datetime.strptime(end_str, "%Y%m%d%H%M") + timedelta(minutes=interval_minutes)

                        # Add days with data to the set
                        current_day = start
                        while current_day <= end:
                            days_with_data.add(current_day.date())
                            current_day += timedelta(days=1)

                        # Plot gap (unavailable period) in red
                        if start > last_end:
                            ax.plot([last_end, start], [0.5, 0.5], color='red', lw=6, solid_capstyle='butt')

                        # Plot continuous period in green
                        ax.plot([start, end], [0.5, 0.5], color='green', lw=6, solid_capstyle='butt')
                        last_end = end
                    except ValueError as e:
                        print(f"Skipping invalid file in period: {period} ({e})")
                        continue
                    pbar.update(1)

                # Plot any final gap after the last period
                if last_end < year_end:
                    ax.plot([last_end, year_end], [0.5, 0.5], color='red', lw=6, solid_capstyle='butt')

            # Set x-axis limits and ticks for each subplot
            ax.set_xlim(year_start, year_end)
            ax.set_xticks([datetime(year, month, 1) for month in range(1, 13)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

            # Ensure y-axis is visible enough to display data
            ax.set_yticks([])
            ax.set_ylim(0, 1)

            # Display the number of days with data in the title
            days_count = len(days_with_data)
            ax.set_title(f"Timeline for {year} ({days_count} days available)", fontsize=12, pad=5)
            ax.grid(visible=True, axis='x', linestyle='--', alpha=0.5)

            # Add the days from this year to the total days count
            total_days_with_data.update(days_with_data)

        # Display total number of available days across all years at the bottom of the figure
        total_days = len(total_days_with_data)
        
        # Set the figure's title to show the total number of days and the file type
        if file_type == "HEL":
            plt.suptitle(f"Hellenic Logger Processed Data (available: {total_days} days)", fontsize=14, fontweight='bold', y=0.98)
        elif file_type == "SRD":
            plt.suptitle(f"Hellenic Logger Original Data (available: {total_days} days)", fontsize=14, fontweight='bold', y=0.98)
        elif file_type == "POL":
            plt.suptitle(f"Polski Logger Processed Data (available: {total_days} days)", fontsize=14, fontweight='bold', y=0.98)
        elif file_type == "DAT":
            plt.suptitle(f"Polski Logger Original Data (available: {total_days} days)", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(pad=3.0)  # Adjust padding to ensure the title doesn't overlap

        # Adjust labels and layout after plotting all years
        plt.xlabel("Months", fontsize=12)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    app = QApplication([])
    window = FileSelectorApp()
    window.show()
    app.exec_()
