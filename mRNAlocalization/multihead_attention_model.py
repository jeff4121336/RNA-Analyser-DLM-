import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from hier_attention_mask import AttentionMask
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
)

# class PostitionEmbedding(nn.Module):


class MultiscaleCNN(nn.Module):
    def __init__(
        self,
        max_len,
        embedding_vec,
        save_path,
        in_channels=64,
        drop_rate_cnn=0.2,
        drop_rate_fc=0.2,
        pooling_size=8,
        pooling_stride=8,
        gelu_used=True,
        nb_classes=6,
    ):
        super(MultiscaleCNN, self).__init__()
        self.max_len = max_len
        self.embedding_vec = embedding_vec
        self.in_channels = in_channels
        self.drop_rate_cnn = drop_rate_cnn
        self.drop_rate_fc = drop_rate_fc
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.gelu_used = gelu_used
        self.nb_classes = nb_classes
        self.save_path = save_path

        # Embedding Mechanism here # Change to one-hot encoding
        # self.embedding = nn.Embedding(
        #     num_embeddings=1000, embedding_dim=128
        # )

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels // 2)

        # CNN1
        self.conv1_1 = nn.Conv1d(
            in_channels=embedding_vec.shape[1],
            out_channels=in_channels,
            kernel_size=9,
            padding="same",
        )
        self.conv1_2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=9,
            padding="same",
        )

        # CNN2
        self.conv2_1 = nn.Conv1d(
            in_channels=embedding_vec.shape[1],
            out_channels=in_channels,
            kernel_size=20,
            padding="same",
        )
        self.conv2_2 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=20,
            padding="same",
        )

        # CNN3
        self.conv3_1 = nn.Conv1d(
            in_channels=embedding_vec.shape[1],
            out_channels=in_channels // 2,
            kernel_size=49,
            padding="same",
        )
        self.conv3_2 = nn.Conv1d(
            in_channels=in_channels // 2,
            out_channels=in_channels // 2,
            kernel_size=49,
            padding="same",
        )

        # Maxpooling layers
        self.pool = nn.MaxPool1d(
            kernel_size=self.pooling_size, stride=self.pooling_stride
        )

        # Head for Multihead Attention
        self.attentionHead = AttentionMask(
			hidden=in_channels // 2,
			da=32,
			r=8,
			returnAttention=False,
			attentionRegularizerWeight=0.0,
			normalize=True,
			attmod="smooth",
			sharpBeta=1
		)

        # Activation functions
        if self.gelu_used:
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        # Dropout layers
        self.dropout_cnn = nn.Dropout(self.drop_rate_cnn)
        self.dropout_fc = nn.Dropout(self.drop_rate_fc)

        # Flatten layer (2D)
        self.flatten = nn.Flatten()

        # Fully connected layer (Edit needed)
        self.fc = nn.Linear(32 * 3, nb_classes)

    def forward(self, x, mask):
        # One-hot encoding (modify needed)
        x = F.one_hot(x, num_classes=4).float()
        x = self.dropout_input(x)
        x = x.permute(0, 2, 1)  # Change to (batch_size, channels, seq_len)

        # CNN1
        x1 = self.activation(self.bn1(self.conv1_1(x)))
        x1 = self.activation(self.bn2(self.conv1_2(x1)))
        x1 = self.pool(x1)
        x1, x1_regularization_loss, x1_attention = self.attentionHead(x1)

        # CNN2
        x2 = self.activation(self.bn1(self.conv2_1(x)))
        x2 = self.activation(self.bn2(self.conv2_2(x2)))
        x2 = self.pool(x2)
        x2, x2_regularization_loss, x2_attention = self.attentionHead(x2)

        # CNN3
        x3 = self.activation(self.bn2(self.conv3_1(x)))
        x3 = self.activation(self.bn2(self.conv3_2(x3)))
        x3 = self.pool(x3)
        x3, x3_regularization_loss, x3_attention = self.attentionHead(x3)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x3 = self.dropout(x3)
 
        x = torch.cat((x1, x2, x3), dim=1) # (batch_size, combined_channels, seq_len)
        x = self.flatten(x) # (batch_size, combined_channels * seq_len)
        x = self.fc(x) # (batch_size, nb_classes)

        return x

    def get_encodings(self, X):
        self.eval()
        with torch.no_grad():
            encoding = self.forward(X, None)
        return encoding

    def get_PCM_multiscale_weighted(
        self,
        X,
        mask_label,
        nb_filters,
        filters_length1,
        filters_length2,
        filters_length3,
    ):
        onehotX = self.get_encodings(X)
        feature_model1 = self.conv1_1
        feature_model2 = self.conv2_1
        feature_model3 = self.conv3_1

        def add(feature_length, up=True):
            if up:
                return int((feature_length - 1) / 2)
            else:
                return feature_length - 1 - int((feature_length - 1) / 2)

        Add1up = add(filters_length1, True)
        Add1down = add(filters_length1, False)
        Add2up = add(filters_length2, True)
        Add2down = add(filters_length2, False)
        Add3up = add(filters_length3, True)
        Add3down = add(filters_length3, False)

        for m in range(nb_filters):
            PCM1 = np.zeros((filters_length1, 4))
            PCM2 = np.zeros((filters_length2, 4))
            PCM3 = np.zeros((filters_length3, 4))
            for s in range(len(X)):
                CNNoutputs1 = feature_model1(X[s : s + 1])
                CNNoutputs2 = feature_model2(X[s : s + 1])
                sub_index1 = CNNoutputs1[0, :, m].argmax() - Add1up
                sub_index2 = CNNoutputs2[0, :, m].argmax() - Add2up
                if m < int(nb_filters / 2):
                    CNNoutputs3 = feature_model3(X[s : s + 1])
                    sub_index3 = CNNoutputs3[0, :, m].argmax() - Add3up

                if CNNoutputs1[0, :, m].max() > 0:
                    if (
                        sub_index1 >= 0
                        and sub_index1 + filters_length1 < onehotX.shape[1]
                    ):
                        PCM1 = (
                            PCM1
                            + onehotX[s, sub_index1 : (sub_index1 + filters_length1), :]
                            * CNNoutputs1[0, :, m].max()
                        )
                    elif sub_index1 < 0:
                        PCM1 = (
                            PCM1
                            + np.pad(
                                onehotX[s, 0 : sub_index1 + filters_length1, :],
                                ([-sub_index1, 0], [0, 0]),
                                "constant",
                                constant_values=0,
                            )
                            * CNNoutputs1[0, :, m].max()
                        )
                    else:
                        PCM1 = (
                            PCM1
                            + np.pad(
                                onehotX[s, sub_index1:, :],
                                (
                                    [
                                        0,
                                        filters_length1 - onehotX.shape[1] + sub_index1,
                                    ],
                                    [0, 0],
                                ),
                                "constant",
                                constant_values=0,
                            )
                            * CNNoutputs1[0, :, m].max()
                        )
                if CNNoutputs2[0, :, m].max() > 0:
                    if (
                        sub_index2 >= 0
                        and sub_index2 + filters_length2 < onehotX.shape[1]
                    ):
                        PCM2 = (
                            PCM2
                            + onehotX[s, sub_index2 : (sub_index2 + filters_length2), :]
                            * CNNoutputs2[0, :, m].max()
                        )
                    elif sub_index2 < 0:
                        PCM2 = (
                            PCM2
                            + np.pad(
                                onehotX[s, 0 : sub_index2 + filters_length2, :],
                                ([-sub_index2, 0], [0, 0]),
                                "constant",
                                constant_values=0,
                            )
                            * CNNoutputs2[0, :, m].max()
                        )
                    else:
                        PCM2 = (
                            PCM2
                            + np.pad(
                                onehotX[s, sub_index2:, :],
                                (
                                    [
                                        0,
                                        filters_length2 - onehotX.shape[1] + sub_index2,
                                    ],
                                    [0, 0],
                                ),
                                "constant",
                                constant_values=0,
                            )
                            * CNNoutputs2[0, :, m].max()
                        )
                if m < int(nb_filters / 2):
                    if CNNoutputs3[0, :, m].max() > 0:
                        if (
                            sub_index3 >= 0
                            and sub_index3 + filters_length3 < onehotX.shape[1]
                        ):
                            PCM3 = (
                                PCM3
                                + onehotX[
                                    s, sub_index3 : (sub_index3 + filters_length3), :
                                ]
                                * CNNoutputs3[0, :, m].max()
                            )
                        elif sub_index3 < 0:
                            PCM3 = (
                                PCM3
                                + np.pad(
                                    onehotX[s, 0 : sub_index3 + filters_length3, :],
                                    ([-sub_index3, 0], [0, 0]),
                                    "constant",
                                    constant_values=0,
                                )
                                * CNNoutputs3[0, :, m].max()
                            )
                        else:
                            PCM3 = (
                                PCM3
                                + np.pad(
                                    onehotX[s, sub_index3:, :],
                                    (
                                        [
                                            0,
                                            filters_length3
                                            - onehotX.shape[1]
                                            + sub_index3,
                                        ],
                                        [0, 0],
                                    ),
                                    "constant",
                                    constant_values=0,
                                )
                                * CNNoutputs3[0, :, m].max()
                            )

            np.savetxt(
                self.save_path + "/PCMmultiscale_weighted_filter1_{}.txt".format(m),
                PCM1,
                delimiter=",",
            )
            np.savetxt(
                self.save_path + "/PCMmultiscale_weighted_filter2_{}.txt".format(m),
                PCM2,
                delimiter=",",
            )
            if m < int(nb_filters / 2):
                np.savetxt(
                    self.save_path + "/PCMmultiscale_weighted_filter3_{}.txt".format(m),
                    PCM3,
                    delimiter=",",
                )

    def get_PCM_multiscale(
        self,
        X,
        mask_label,
        nb_filters,
        filters_length1,
        filters_length2,
        filters_length3,
    ):
        onehotX = self.get_encodings(X)
        feature_model1 = self.conv1_1
        feature_model2 = self.conv2_1
        feature_model3 = self.conv3_1

        def add(feature_length, up=True):
            if up:
                return int((feature_length - 1) / 2)
            else:
                return feature_length - 1 - int((feature_length - 1) / 2)

        Add1up = add(filters_length1, True)
        Add1down = add(filters_length1, False)
        Add2up = add(filters_length2, True)
        Add2down = add(filters_length2, False)
        Add3up = add(filters_length3, True)
        Add3down = add(filters_length3, False)

        for m in range(nb_filters):
            PCM1 = np.zeros((filters_length1, 4))
            PCM2 = np.zeros((filters_length2, 4))
            PCM3 = np.zeros((filters_length3, 4))
            for s in range(len(X)):
                CNNoutputs1 = feature_model1(X[s : s + 1])
                CNNoutputs2 = feature_model2(X[s : s + 1])
                sub_index1 = CNNoutputs1[0, :, m].argmax() - Add1up
                sub_index2 = CNNoutputs2[0, :, m].argmax() - Add2up
                if m < int(nb_filters / 2):
                    CNNoutputs3 = feature_model3(X[s : s + 1])
                    sub_index3 = CNNoutputs3[0, :, m].argmax() - Add3up

                if CNNoutputs1[0, :, m].max() > 0:
                    if (
                        sub_index1 >= 0
                        and sub_index1 + filters_length1 < onehotX.shape[1]
                    ):
                        PCM1 = (
                            PCM1
                            + onehotX[s, sub_index1 : (sub_index1 + filters_length1), :]
                        )
                    elif sub_index1 < 0:
                        PCM1 = PCM1 + np.pad(
                            onehotX[s, 0 : sub_index1 + filters_length1, :],
                            ([-sub_index1, 0], [0, 0]),
                            "constant",
                            constant_values=0,
                        )
                    else:
                        PCM1 = PCM1 + np.pad(
                            onehotX[s, sub_index1:, :],
                            (
                                [0, filters_length1 - onehotX.shape[1] + sub_index1],
                                [0, 0],
                            ),
                            "constant",
                            constant_values=0,
                        )
                if CNNoutputs2[0, :, m].max() > 0:
                    if (
                        sub_index2 >= 0
                        and sub_index2 + filters_length2 < onehotX.shape[1]
                    ):
                        PCM2 = (
                            PCM2
                            + onehotX[s, sub_index2 : (sub_index2 + filters_length2), :]
                        )
                    elif sub_index2 < 0:
                        PCM2 = PCM2 + np.pad(
                            onehotX[s, 0 : sub_index2 + filters_length2, :],
                            ([-sub_index2, 0], [0, 0]),
                            "constant",
                            constant_values=0,
                        )
                    else:
                        PCM2 = PCM2 + np.pad(
                            onehotX[s, sub_index2:, :],
                            (
                                [0, filters_length2 - onehotX.shape[1] + sub_index2],
                                [0, 0],
                            ),
                            "constant",
                            constant_values=0,
                        )
                if m < int(nb_filters / 2):
                    if CNNoutputs3[0, :, m].max() > 0:
                        if (
                            sub_index3 >= 0
                            and sub_index3 + filters_length3 < onehotX.shape[1]
                        ):
                            PCM3 = (
                                PCM3
                                + onehotX[
                                    s, sub_index3 : (sub_index3 + filters_length3), :
                                ]
                            )
                        elif sub_index3 < 0:
                            PCM3 = PCM3 + np.pad(
                                onehotX[s, 0 : sub_index3 + filters_length3, :],
                                ([-sub_index3, 0], [0, 0]),
                                "constant",
                                constant_values=0,
                            )
                        else:
                            PCM3 = PCM3 + np.pad(
                                onehotX[s, sub_index3:, :],
                                (
                                    [
                                        0,
                                        filters_length3 - onehotX.shape[1] + sub_index3,
                                    ],
                                    [0, 0],
                                ),
                                "constant",
                                constant_values=0,
                            )

            np.savetxt(
                self.save_path + "/PCMmultiscale_filter1_{}.txt".format(m),
                PCM1,
                delimiter=",",
            )
            np.savetxt(
                self.save_path + "/PCMmultiscale_filter2_{}.txt".format(m),
                PCM2,
                delimiter=",",
            )
            if m < int(nb_filters / 2):
                np.savetxt(
                    self.save_path + "/PCMmultiscale_filter3_{}.txt".format(m),
                    PCM3,
                    delimiter=",",
                )


# Example usage:
# model = MultiheadAttentionModel(max_len=100, nb_classes=10, save_path='/path/to/save', kfold_index=1)
# output = model(torch.randint(0, 1000, (32, 100)), None)
