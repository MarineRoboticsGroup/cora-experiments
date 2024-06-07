import os
import shutil

def modify_tum_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.tum') and "cora" in filename:
            print(f"Modifying {filename}" )
            input_file = os.path.join(directory, filename)
            backup_file = input_file + '.orig'
            output_file = input_file

            # Create a backup of the original file
            shutil.copy(input_file, backup_file)

            # Modify the file
            with open(input_file, 'r') as infile, open(output_file + '.tmp', 'w') as outfile:
                for line_number, line in enumerate(infile, start=1):
                    parts = line.split()
                    if len(parts) > 1:
                        # rewrite timestamp with line number
                        parts[0] = str(line_number)

                        # flip the sign of qz (z-coordinate of the quaternion)
                        if "cora" in filename:
                            parts[7] = str(-float(parts[7]))

                        outfile.write(' '.join(parts) + '\n')

            # Replace the original file with the modified file
            os.rename(output_file + '.tmp', output_file)

# Directory containing the .tum files
for i in [2, 4, 7]:
    directory = f'/home/alan/range-only-slam-mission-control/cora-experiments/data/mrclam{i}'
    modify_tum_files_in_directory(directory)
