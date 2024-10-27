import pandas as pd
import re

def original():
	combined_data = pd.concat([pd.read_csv("./train.csv"), pd.read_csv("./test.csv"), pd.read_csv("./dev.csv")], ignore_index=True) # Combined data from train.csv and dev.csv
	shuffled_df = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)
	shuffled_df.to_csv("original.csv", index=False)

def balanced():
	dataset = pd.read_csv('original.csv')
	label_1_data = dataset[dataset['labels'] == 1]
	label_0_data = dataset[dataset['labels'] == 0].sample(n=len(label_1_data))
	combined_data = pd.concat([label_1_data, label_0_data]).sample(frac=1)
	combined_data.to_csv('5050ds2.csv', index=False)

def strange(dtype, d):
	dataset = pd.read_csv('original.csv')
	if dtype == "exon":
		data = dataset[dataset['labels'] == 1].sample(n=d, random_state=42)
		data.to_csv('allexon.csv', index=False)
	elif dtype == "intron":
		data = dataset[dataset['labels'] == 0].sample(n=d, random_state=42)
		data.to_csv('allintron.csv', index=False)
	
def newdata(file_path):
	with open(file_path, 'r') as file:
		content = file.read()

	lines = content.split('\n')
	updated_lines = []
	for line in lines:
		if '>' in line:
			updated_lines.append('\n' + line + '\n')
		else:
			updated_lines.append(re.sub(r'[\r\n]+', '', line))

	updated_content = ''.join(updated_lines)

	with open(file_path, 'w') as file:
		file.write(updated_content)

	with open(file_path, 'r') as file:
		lines = file.readlines()

	data = []
	print(len(lines))
	for i in range(1, len(lines), 2):
		data_type = 0 if "intron" in lines[i].strip() else 1
		data_lines = lines[i + 1].strip().split(' ')
		for line in data_lines:
			if len(line) > 200:
				# Split long data line into chunks of 200 characters
				chunks = [line[j:j+200] for j in range(0, len(line), 200)]
				for chunk in chunks:
					data.append({'Data': chunk, 'Type': data_type })
			else:
				data.append({'Data': line, 'Type': data_type })

	df = pd.DataFrame(data)

	df.to_csv("new_data.csv")
	
	return 
      
if __name__ == "__main__":
	# balanced()
	# strange('intron', 80) #cap at 87
	newdata("ft1_combined.fasta")
	