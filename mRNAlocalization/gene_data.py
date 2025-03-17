import numpy as np

np.random.seed(1234)

class Genedata:
    trainTestSplitRatio = 0.1

    def __init__(self, id, label):
        self.id = id
        self.label = label
        self.seq = None
        self.seqLeft = None
        self.seqRight = None
        self.length = None

    @staticmethod
    def create_gene(id, label, seq, left, right):
        #
        # Creating gene object from sequence
        # Translate, Trim, Details extraction of the sequence
        # Retrun gene object
        #
        seq = seq.upper()
        seq = seq.replace("U", "T")
        seq = list(seq)
        for seqIndex in range(len(seq)):
            if seq[seqIndex] not in ["A", "C", "G", "T"]:
                seq[seqIndex] = "N"
        seq = "".join(seq)
        seqLength = len(seq)
        lineLeft = seq[: int(seqLength * left / (left + right))]
        lineRight = seq[int(seqLength * left / (left + right)) :]

        if len(lineLeft) >= left:
            lineLeft = lineLeft[:left]
        if len(lineRight) >= right:
            lineRight = lineRight[-right:]

        gene = Genedata(id, label)
        gene.seqLeft = lineLeft.rstrip()
        gene.seqRight = lineRight.rstrip()
        gene.length = seqLength
        return gene

    @classmethod
    def load_sequence(cls, dataset, left=1000, right=1000, predict=False):
        #
        # Load the dataset and create gene training and testing dataset
        # Reading files and creating gene object
        # Return the gene dataset (genes)
        #
        genes = []
        path = dataset
        print("Importing dataset {0}".format(dataset))

        with open(path, "r") as f:
            seqIndex = 0
            for line in f:
                if line[0] == ">":
                    if seqIndex > 0:
                        gene = cls.create_gene(id, label, seq, left, right)
                        genes.append(gene)

                    id = line.strip()
                    label = line[1:].split(",")[0]
                    seq = ""
                else:
                    seq += line.strip()
                seqIndex += 1

            # last sequence
            gene = cls.create_gene(id, label, seq, left, right)
            genes.append(gene)

        genes = np.array(genes)

        if not predict:
            genes = genes[np.random.permutation(np.arange(len(genes)))]

        print("Total number of genes: {0}".format(genes.shape[0]))
        return genes
