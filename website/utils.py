import torch
import fm
import numpy as np
#hyperparameters
labels_ref = {"5S_rRNA": 0, "5_8S_rRNA": 1, "tRNA": 2, "ribozyme": 3, "CD-box": 4, "miRNA": 5,
		"Intron_gpI": 6, "Intron_gpII": 7,  "scaRNA": 8, "HACA-box": 9, "riboswitch": 10, "IRES": 11, "leader": 12, "unknown": 13, "pad": 14}
reverse_labels_ref = {v: k for k, v in labels_ref.items()}
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
padding_token = '-'

def crop_sequences(sequences_labels, segment_length=32, overlap=12):
	"""
		Crop input RNA into smaller parts for training and features mapping purpose.
		Allow list of sequences.

		return segments
	"""
	# example data in sequences_labels,
	# (('RF00168_ACCK01000036_1_8201-8035', 'UAUAAAGAUAGAUGUUGUCUUGGUGAUUCAGGACAUGAUGAAGAGGCGUUAUCUUAUCG'), 10)
	segments = [[] for _ in range(len(sequences_labels))]
	sequence = [seq[1] for seq, _ in sequences_labels]
	labels = [lb for _, lb in sequences_labels]

	for i, seq in enumerate (sequence):
		# print(f"Processing sequence {i}: {seq}")
		# seq = ''.join(maps[nucleotide] for nucleotide in seq)
		for j in range(0, len(seq), segment_length - overlap):
			if j + segment_length >= len(seq):
				remaining_segment = seq[j:len(seq)]
				padded_segment = remaining_segment + (padding_token * (segment_length-len(remaining_segment)))
				segments[i].append((padded_segment, labels[i])) 
				# print(f"Segment (_if_) added: {segments[i]}")
				break
			else:
				segments[i].append((seq[j :(j+segment_length)], labels[i]))
				# print(f"Segment (else) added: {segments[i]}")
			# time.sleep(1)
	return segments


def prepare(sequence):
	sequence = ''.join({'T' : 'U'}.get(base, base) for base in sequence) 
	# print(sequence)
	data = [[("RNA1", sequence), 99]]
	crop_seq = crop_sequences(data)
	model, alphabet = fm.pretrained.rna_fm_t12()
	batch_converter = alphabet.get_batch_converter()
	model.to(device)
	# print(len(segments[0]))
	segments = []
	for i, seg in enumerate(crop_seq[0]):
		segments.append(seg[0])

	segment_tuple = []
	segment_batch_size = 30
	segment_length = 32
	length_wo_padding = []

	maximum_length = segment_batch_size * segment_length
	for k in range(0, len(segments), segment_batch_size):
		segment = ''.join(segments[k:k + segment_batch_size])
		length_wo_padding.append(len(segment) // segment_length)
		sequence_id = f"Sequence {i} Segment {k} to {min(k + segment_batch_size, len(segments)) - 1}"
		segment_padding = '-' * (maximum_length - len(segment))
		segment = segment + segment_padding
		segment_tuple.append((sequence_id, segment))

	batch_labels, batch_strs, batch_tokens = batch_converter(segment_tuple)

	with torch.no_grad():
		results = model(batch_tokens.to(device), repr_layers=[12])
		emb = results['representations'][12].cpu().numpy() 

	token_embeddings = []
	for j in range(emb.shape[0]):
		token_embeddings.append(emb[j:j+1])
	token_embeddings = np.concatenate(token_embeddings, axis=0)
	token_embeddings = token_embeddings[:, 1:-1, :]

	# print(token_embeddings.shape)
	temp = []
	mean_idx = [i for i in range(0, token_embeddings.shape[1], 32)]
	for i in range(token_embeddings.shape[0]):
		for j in range(len(mean_idx)):
			temp.append(np.mean(token_embeddings[i][mean_idx[j]: mean_idx[j]+32], axis=0))
		# Use the mean of the RNA-FM embedding across 32 items
	token_embeddings = torch.tensor(np.array(temp))
	x = token_embeddings.to(device).float()
	x = x.view(-1, 30, 640)

	target_labels = []
	class_label = -1
	for i in range(len(length_wo_padding)):
		target_labels.append([class_label] * segment_batch_size)

	target_tensor = torch.tensor(target_labels)
	target_tensor = target_tensor.view(x.shape[0], 30, ).to(device)

	return x, target_tensor