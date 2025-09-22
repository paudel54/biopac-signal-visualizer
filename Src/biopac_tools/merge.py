from datetime import datetime, timedelta 
import pandas as pd
from io import StringIO

#parse biopac files 

def parse_biopac_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines() 
    #find header dynamically, current idx 15. idx starts from 0
    hdr_idx = next(i for i , line in enumerate(lines) if line.startswith("sec,"))
    #parse headers before lines starting from sec, i.e microsiemens included.
    header = lines[:hdr_idx]
    #Extract recording start time from header 
    rec_line = next(line for line in header if "Recording on:" in line)
    start_str = rec_line.split("Recording on:")[1].strip()
    start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S.%f")
    #Data line 
    data_lines = lines[hdr_idx + 2:]
    df = pd.read_csv(StringIO("".join(data_lines)), header=None, usecols=range(6))
    df.columns = ["sec", "CH1", "CH2", "CH3", "CH7", "CH16"]

    #Generate the realtive timestamp into actual timestamps
    df["timestamp"] = start_time + pd.to_timedelta(df["sec"], unit="s")
    print(df)
    return header, df 

# def merge_chunks_with_gap(chunks,sample_rate=1000):
    # #Combine all chunks. 
    # #chunks = [(header, df1), (header,df2),....]
    # all_df = pd.concat(df for _, df in chunks).sort_values("timestamp").reset_index(drop=True)
    # start, end = all_df["timestamp"].iloc[[0,-1]] 
    # #Generate complete timestamp index at 1ms intervals
    # full_index = pd.date_range(start=start, end=end, freq=f"{1000/sample_rate}ms") 
    # merged = pd.DataFrame({"timestamp": full_index})
    # #Merge and fill gaps with zeros
    # merged = pd.merge(merged, all_df, on="timestamp", how="left").fillna(0.0)
    # #Recompute 'sec' based on index position 
    # merged["sec"] = (merged.index/sample_rate).round(3)
    # return merged
def merge_chunks_with_gap(chunks, sample_rate=1000):
    merged_chunks = []
    current_time = None
    total_samples = 0  # To track and build 'sec' column

    for idx, (header, df) in enumerate(chunks):
        df = df.copy()

        # Determine start and end time of this chunk
        chunk_start = df["timestamp"].iloc[0]
        chunk_end = df["timestamp"].iloc[-1] + timedelta(seconds=1/sample_rate)

        if current_time is not None:
            # Calculate time gap
            gap_duration = (chunk_start - current_time).total_seconds()
            gap_samples = int(round(gap_duration * sample_rate))

            if gap_samples > 0:
                # Create gap rows with 0s
                print(f"Inserted gap of {gap_samples} samples ({gap_duration:.3f} seconds) between chunk {idx-1} and chunk {idx}")
                gap_data = {
                    "sec": [total_samples + i / sample_rate for i in range(gap_samples)],
                    "CH1": [0.0] * gap_samples,
                    "CH2": [0.0] * gap_samples,
                    "CH3": [0.0] * gap_samples,
                    "CH7": [0.0] * gap_samples,
                    "CH16": [0.0] * gap_samples,
                }
                gap_df = pd.DataFrame(gap_data)
                gap_df["timestamp"] = [current_time + timedelta(seconds=i / sample_rate) for i in range(gap_samples)]
                merged_chunks.append(gap_df)
                total_samples += gap_samples

        # Adjust sec for this chunk to be continuous
        df["sec"] = [total_samples + i / sample_rate for i in range(len(df))]
        total_samples += len(df)

        merged_chunks.append(df)
        current_time = chunk_end

    # Final merge
    merged_df = pd.concat(merged_chunks).reset_index(drop=True)
    return merged_df


def write_merged_biopac_file(output_path,header,df):
    sample_count = len(df)
    channels = ["CH1", "CH2", "CH3", "CH7", "CH16"] 
    sample_line = "," + ",".join(f"{sample_count} samples" for _ in channels) + ",\n"
    with open(output_path, 'w') as f:
        f.writelines(header) 
        f.write("sec,CH1,CH2,CH3,CH7,CH16,\n")
        f.write(sample_line)
        for _, row in df.iterrows():
            line = f"{row['sec']},{row['CH1']},{row['CH2']},{row['CH3']},{row['CH7']},{row['CH16']}\n"
            f.write(line)
