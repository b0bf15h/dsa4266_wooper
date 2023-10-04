import pandas as pd
import json
import gzip
import os

def find_keys(d):
    keys = []
    for key, value in d.items():
        # recursively adds keys to list
        if isinstance(value, dict):
          keys.append(key)
          keys.extend(find_keys(value))
        else:
          keys.append(key)
    return keys

def parse_seq_pos0_only (d):
  '''For each read of 9 data points, return lists of their info '''
  keys = find_keys(d)
  # extract info from each set of keys including adjacent positions
  transcript_id = keys[0]
  position = keys[1]
  sequences = [keys[2][0:5],keys[2][1:6],keys[2][2:7]]
  reads = d[keys[0]][keys[1]][keys[2]]
  transcript_id = [transcript_id]*len(reads)
  seq_pos, seq_seq, seq_dtime, seq_sd, seq_mean = [],[],[],[],[]
  m1_seq, m1_dtime, m1_sd, m1_mean = [],[],[],[]
  p1_seq, p1_dtime, p1_sd, p1_mean = [],[],[],[]
  for read in reads:
    seq_pos.append(position)
    seq_seq.append(sequences[1])
    seq_dtime.append(read[3])
    seq_sd.append(read[4])
    seq_mean.append(read[5])
    m1_seq.append(sequences[0])
    m1_dtime.append(read[0])
    m1_sd.append(read[1])
    m1_mean.append(read[2])
    p1_seq.append(sequences[2])
    p1_dtime.append(read[6])
    p1_sd.append(read[7])
    p1_mean.append(read[8])
  return transcript_id, seq_pos, seq_seq, seq_dtime, seq_sd, seq_mean, m1_seq, m1_dtime, m1_sd, m1_mean, p1_seq, p1_dtime, p1_sd, p1_mean

def parse_data_pos0(lis):
  '''Iterate through data and create the lists '''
  transcript_id, seq_pos, seq_seq, seq_dtime, seq_sd, seq_mean, m1_seq, m1_dtime, m1_sd, m1_mean, p1_seq, p1_dtime, p1_sd, p1_mean = [],[],[],[],[],[],[],[],[],[],[],[],[],[]
  for entry in data:
    t_id, s_pos, s_seq, s_dtime, s_sd, s_mean, m_seq, m_dtime, m_sd, m_mean, p_seq, p_dtime, p_sd, p_mean = parse_seq_pos0_only(entry)
    transcript_id.extend(t_id)
    seq_pos.extend(s_pos)
    seq_seq.extend(s_seq)
    seq_dtime.extend(s_dtime)
    seq_sd.extend(s_sd)
    seq_mean.extend(s_mean)
    m1_seq.extend(m_seq)
    m1_dtime.extend(m_dtime)
    m1_sd.extend(m_sd)
    m1_mean.extend(m_mean)
    p1_seq.extend(p_seq)
    p1_dtime.extend(p_dtime)
    p1_sd.extend(p_sd)
    p1_mean.extend(p_mean)
  return transcript_id, seq_pos, seq_seq, seq_dtime, seq_sd, seq_mean, m1_seq, m1_dtime, m1_sd, m1_mean, p1_seq, p1_dtime, p1_sd, p1_mean

# via GPT3.5 'decompressing a json.gz file into a json file'
input_file = 'dataset0.json.gz'
output_file = 'data.json'

# Open the compressed file and the output file
with gzip.open(input_file, 'rb') as gzipped_file:
    with open(output_file, 'wb') as json_file:
        # Read the compressed data and write it to the output file
        json_data = gzipped_file.read()
        json_file.write(json_data)
        
data = []
with open('data.json', 'r') as file:
    for line in file:
        data.append(json.loads(line))

os.remove('data.json')
transcript_id, seq_pos, seq_seq, seq_dtime, seq_sd, seq_mean, m1_seq, m1_dtime, m1_sd, m1_mean, p1_seq, p1_dtime, p1_sd, p1_mean = parse_data_pos0(data)

df = pd.DataFrame({'transcript_id':transcript_id, 'position':seq_pos, 'sequence':seq_seq, 'dwell_time': seq_dtime, 'sd':seq_sd, 'mean': seq_mean, 'm1_seq':m1_seq, 'm1_dtime': m1_dtime,
                        'm1_sd': m1_sd, 'm1_mean': m1_mean, 'p1_seq': p1_seq, 'p1_dtime' : p1_dtime, 'p1_sd': p1_sd, 'p1_mean' : p1_mean})

# GPT3.5 in python, replace all occurrences of a character from a given string using lambda function
seq = df['sequence']
new_seq = []
for s in seq:
  result = ''.join(map(lambda x: x if x != 'T' else 'U', s))
  new_seq.append(result)
df['sequence'] = new_seq


m1_seq = df['m1_seq']
new_seq = []
for s in m1_seq:
  result = ''.join(map(lambda x: x if x != 'T' else 'U', s))
  new_seq.append(result)
df['m1_seq'] = new_seq

p1_seq = df['p1_seq']
new_seq = []
for s in p1_seq:
  result = ''.join(map(lambda x: x if x != 'T' else 'U', s))
  new_seq.append(result)
df['p1_seq'] = new_seq

df.to_pickle('unlabelled_data.pkl')