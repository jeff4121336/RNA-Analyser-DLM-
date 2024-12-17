import glob
from pathlib import Path
from Bio import SeqIO
from collections import Counter
import random

random.seed(42)
def get_data(filename):

  labels_ref = {"5S_rRNA": 0, "5_8S_rRNA": 1, "tRNA": 2, "ribozyme": 3, "CD-box": 4, "miRNA": 5,
            "Intron_gpI": 6, "Intron_gpII": 7,  "scaRNA": 8, "HACA-box": 9, "riboswitch": 10, "IRES": 11, "leader": 12,
            "mRNA": 13, "unknown": 14}
  reverse_labels_ref = {v: k for k, v in labels_ref.items()}
  
  ret = []
  seqs = []
  labels = [] 
  print(f"Processing Files: {filename}")
  
  records = list(SeqIO.parse(filename, 'fasta'))
  
  for record in records:
    sequence_desc = record.description  # Get the definition line (sequence ID)
    sequence_id = record.id
    sequence = map_rna(str(record.seq))  # Get the sequence
    
    sequence_desc = sequence_desc.split()[-1]
    label = labels_ref.get(filename, labels_ref["unknown"])
    for key in labels_ref.keys():
      if key == sequence_desc:
        label = labels_ref[key] 
        # print(label)
        break  # Exit loop once a match is found

    seqs.append((sequence_id, sequence))
    labels.append(label)
  
  for i in range(len(labels)):
    ret.append([(seqs[i][0],seqs[i][1]), labels[i]])
  
  label_counts = Counter(labels)
  for label, count in label_counts.items():
    print(f"{reverse_labels_ref[label]}: {count} ")
      
  return ret 


def group_data():
  
  train_data_dir = glob.glob(r'./data/fa_train*')
  test_data_dir = glob.glob(r'./data/fa_test*')
  val_data_dir = glob.glob(r'./data/fa_val*')
  # data_dir = train_data_dir + val_data_dir + test_data_dir
  # print(data_dir)
  train_data = get_data(train_data_dir[0])
  test_data = get_data(test_data_dir[0])
  val_data = get_data(val_data_dir[0])

  return [train_data, test_data, val_data]

def map_rna(sequence):
  mapp = {'T' : 'U'}
  mapped_sequence = ''.join(mapp.get(base, base) for base in sequence)  
  return mapped_sequence


if __name__ == "__main__":
  group_data()
  