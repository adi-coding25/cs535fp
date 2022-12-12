
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_helper import model_helper


class logical_arch_conv(nn.Module):
    def __init__(self):
        super(logical_arch_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 16, 16 * 4 * 4)


class logical_cnn(model_helper):
    def __init__(self, args):
        super(logical_cnn, self).__init__(args)
        self.conv_module = logical_arch_conv()
        self.dimension = 16 * 4 * 4
        self.ans = torch.nn.Parameter(
            torch.from_numpy(
                np.random.uniform(0, 0.1, size=self.dimension).astype(np.float32)
            ),
            requires_grad=False,
        )
        self.sftmx_layer = nn.Softmax(dim=1)

        self.layer1_not = torch.nn.Linear(self.dimension, self.dimension)
        self.layer2_not = torch.nn.Linear(self.dimension, self.dimension)

        self.layer1_and = torch.nn.Linear(2 * self.dimension, self.dimension)
        self.layer2_and = torch.nn.Linear(self.dimension, self.dimension)

        self.layer_dropout = torch.nn.Dropout(0.1)

        # Layers of create_row network
        self.layer1_row = torch.nn.Linear(3 * self.dimension, self.dimension)
        self.layer2_row = torch.nn.Linear(self.dimension, self.dimension)
        self.layer_dropout_row = torch.nn.Dropout(0.1)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.epsilon,
        )

    def gen_row(self, vector1, vector2, vector3, dim=1):
        new_vec = torch.cat((vector1, vector2, vector3), dim)
        new_vec = F.relu(self.layer1_row(new_vec))
        if not self.training:
            pass
        else:
            new_vec = self.layer_dropout_row(new_vec)
        res = self.layer2_row(new_vec)
        return res

    def MSE(self, v_1, v_2):
        v_1, v_2 = self.uniform_size(v_1, v_2, train=False)
        return (v_1 - v_2) ** 2

    def inner_product(self, v_1, v_2):
        v_1, v_2 = self.uniform_size(v_1, v_2, train=False)
        res1 = (v_1 * v_2).sum(dim=-1)
        v1_new = v_1.pow(2).sum(dim=-1).pow(self.sim_alpha)
        v2_new = v_2.pow(2).sum(dim=-1).pow(self.sim_alpha)
        res1 = res1 / torch.clamp(v1_new * v2_new, min=1e-8)
        return res1

    def cosine_similarity(self, vector1, vector2, sigmoid=False):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * 100
        if not sigmoid:
            return result
        return result.sigmoid()

    def logical_NOT(self, new_vec):
        new_vec = F.relu(self.layer1_not(new_vec))
        if not self.training:
            pass
        else:
            new_vec = self.layer_dropout(new_vec)
        res = self.layer2_not(new_vec)
        return res

    def logical_OR(self, vector1, vector2, dim=1):
        new_vec = torch.cat((vector1, vector2), dim)
        new_vec = F.relu(self.layer1_and(new_vec))
        if not self.training:
            pass
        else:
            new_vec = self.layer_dropout(new_vec)
        res1 = self.layer2_and(new_vec)
        return res1

    def logical_AND(self, vector1, vector2, dim=1):
        new_vec = torch.cat((vector1, vector2), dim)
        new_vec = F.relu(self.layer1_and(new_vec))
        if not self.training:
            pass
        else:
            new_vec = self.layer_dropout(new_vec)
        res = self.layer2_and(new_vec)
        return res

    def loss_func(self, output, target, _):
        predictions, constraints, true_constraints = output[0], output[1], output[2]
        false_vector = self.logical_NOT(self.ans)
        r_not_not_true = 1 - F.cosine_similarity(
            self.logical_NOT(self.logical_NOT(self.ans)),
            self.ans,
            dim=0,
        )
        r_not_not_self = (
            1
            - F.cosine_similarity(
                self.logical_NOT(self.logical_NOT(constraints)), constraints
            )
        ).mean()
        r_not_self = (
            1 + F.cosine_similarity(self.logical_NOT(constraints), constraints)
        ).mean()
        r_not_not_not = (
            1
            + F.cosine_similarity(
                self.logical_NOT(self.logical_NOT(constraints)),
                self.logical_NOT(constraints),
            )
        ).mean()
        r_or_true = (
            1
            - F.cosine_similarity(
                self.logical_OR(constraints, self.ans.expand_as(constraints)),
                self.ans.expand_as(constraints),
            )
        ).mean()
        r_or_false = (
            1
            - F.cosine_similarity(
                self.logical_OR(constraints, false_vector.expand_as(constraints)),
                constraints,
            )
        ).mean()

        r_or_self = (
            1
            - F.cosine_similarity(
                self.logical_OR(constraints, constraints), constraints
            )
        ).mean()

        r_or_not_self = (
            1
            - F.cosine_similarity(
                self.logical_OR(constraints, self.logical_NOT(constraints)),
                self.ans.expand_as(constraints),
            )
        ).mean()
        r_or_not_self_inverse = (
            1
            - F.cosine_similarity(
                self.logical_OR(self.logical_NOT(constraints), constraints),
                self.ans.expand_as(constraints),
            )
        ).mean()

        r_and_true = (
            1
            - F.cosine_similarity(
                self.logical_AND(constraints, self.ans.expand_as(constraints)),
                constraints,
            )
        ).mean()

        r_and_false = (
            1
            - F.cosine_similarity(
                self.logical_AND(constraints, false_vector.expand_as(constraints)),
                false_vector.expand_as(constraints),
            )
        ).mean()

        r_and_self = (
            1
            - F.cosine_similarity(
                self.logical_AND(constraints, constraints), constraints
            )
        ).mean()

        r_and_not_self = (
            1
            - F.cosine_similarity(
                self.logical_AND(constraints, self.logical_NOT(constraints)),
                false_vector.expand_as(constraints),
            )
        ).mean()

        r_and_not_self_inverse = (
            1
            - F.cosine_similarity(
                self.logical_AND(self.logical_NOT(constraints), constraints),
                false_vector.expand_as(constraints),
            )
        ).mean()

        output_bool = 1 + F.cosine_similarity(self.ans, false_vector, dim=0)

        r_loss = (
            r_not_not_true
            + r_not_not_self
            + r_not_self
            + r_not_not_not
            + r_or_true
            + r_or_false
            + r_or_self
            + r_or_not_self
            + r_or_not_self_inverse
            + output_bool
            + r_and_true
            + r_and_false
            + r_and_self
            + r_and_not_self
            + r_and_not_self_inverse
        )

        rows_and_true = (
            1
            - F.cosine_similarity(
                self.ans.expand_as(true_constraints), true_constraints
            )
        ).mean()

        loss = F.cross_entropy(predictions, target) + r_loss * 0.1 + rows_and_true * 0.1
        return loss

    def forward(self, x):
        limits_new,limits = [],[]
        image_features = self.conv(x.view(-1, 1, 80, 80))
        choice1, choice2, choice3, choice4, choice5, choice6, choice7, choice8 = self.update_limit_vecs(image_features,
                                                                                                        limits,
                                                                                                        limits_new)
        if len(limits) <= 0:
            pass
        else:
            limits = torch.cat(limits, dim=0)
            limits = limits.view(-1, self.dimension)
            limits_new = torch.cat(limits_new, dim=0)
            limits_new = limits_new.view(-1, self.dimension)
        prediction_c1, prediction_c2, prediction_c3, prediction_c4, prediction_c5, prediction_c6, prediction_c7, prediction_c8 = self.update_preds(
            choice1, choice2, choice3, choice4, choice5, choice6, choice7, choice8)

        score = torch.stack(
            (
                prediction_c1,
                prediction_c2,
                prediction_c3,
                prediction_c4,
                prediction_c5,
                prediction_c6,
                prediction_c7,
                prediction_c8,
            ),
            -1,
        )
        return score, limits, limits_new

    def update_preds(self, choice1, choice2, choice3, choice4, choice5, choice6, choice7, choice8):
        prediction_c1 = self.cosine_similarity(choice1, self.ans).view([-1])
        prediction_c2 = self.cosine_similarity(choice2, self.ans).view([-1])
        prediction_c3 = self.cosine_similarity(choice3, self.ans).view([-1])
        prediction_c4 = self.cosine_similarity(choice4, self.ans).view([-1])
        prediction_c5 = self.cosine_similarity(choice5, self.ans).view([-1])
        prediction_c6 = self.cosine_similarity(choice6, self.ans).view([-1])
        prediction_c7 = self.cosine_similarity(choice7, self.ans).view([-1])
        prediction_c8 = self.cosine_similarity(choice8, self.ans).view([-1])
        return prediction_c1, prediction_c2, prediction_c3, prediction_c4, prediction_c5, prediction_c6, prediction_c7, prediction_c8

    def update_limit_vecs(self, image_features, limits, limits_new):
        row1 = self.gen_row(
            image_features[:, 0], image_features[:, 1], image_features[:, 2]
        )
        limits.append(row1)
        limits_new.append(row1)
        row2 = self.gen_row(
            image_features[:, 3], image_features[:, 4], image_features[:, 5]
        )
        limits.append(row2)
        limits_new.append(row2)
        row1_row2 = self.logical_AND(row1, row2)
        limits.append(row1_row2)
        limits_new.append(row1_row2)
        n_row1_row2_or = self.logical_OR(self.logical_NOT(row1), self.logical_NOT(row2))
        limits.append(n_row1_row2_or)
        row3_1 = self.gen_row(
            image_features[:, 6], image_features[:, 7], image_features[:, 8]
        )
        row3_1_n = self.logical_NOT(row3_1)
        choice1 = self.logical_OR(n_row1_row2_or, row3_1)
        limits.append(row3_1)
        limits.append(row3_1_n)
        limits.append(choice1)
        row3_2 = self.gen_row(
            image_features[:, 6], image_features[:, 7], image_features[:, 9]
        )
        row3_2_n = self.logical_NOT(row3_2)
        choice2 = self.logical_OR(n_row1_row2_or, row3_2)
        limits.append(row3_2)
        limits.append(row3_2_n)
        limits.append(choice2)
        row3_3 = self.gen_row(
            image_features[:, 6], image_features[:, 7], image_features[:, 10]
        )
        row3_3_n = self.logical_NOT(row3_3)
        choice3 = self.logical_OR(n_row1_row2_or, row3_3)
        limits.append(row3_3)
        limits.append(row3_3_n)
        limits.append(choice3)
        row3_4 = self.gen_row(
            image_features[:, 6], image_features[:, 7], image_features[:, 11]
        )
        row3_4_n = self.logical_NOT(row3_4)
        choice4 = self.logical_OR(n_row1_row2_or, row3_4)
        limits.append(row3_4)
        limits.append(row3_4_n)
        limits.append(choice4)
        row3_5 = self.gen_row(
            image_features[:, 6], image_features[:, 7], image_features[:, 12]
        )
        row3_5_n = self.logical_NOT(row3_5)
        choice5 = self.logical_OR(n_row1_row2_or, row3_5)
        limits.append(row3_5)
        limits.append(row3_5_n)
        limits.append(choice5)
        row3_6 = self.gen_row(
            image_features[:, 6], image_features[:, 7], image_features[:, 13]
        )
        row3_6_n = self.logical_NOT(row3_6)
        choice6 = self.logical_OR(n_row1_row2_or, row3_6)
        limits.append(row3_6)
        limits.append(row3_6_n)
        limits.append(choice6)
        row3_7 = self.gen_row(
            image_features[:, 6], image_features[:, 7], image_features[:, 14]
        )
        row3_7_n = self.logical_NOT(row3_7)
        choice7 = self.logical_OR(n_row1_row2_or, row3_7)
        limits.append(row3_7)
        limits.append(row3_7_n)
        limits.append(choice7)
        row3_8 = self.gen_row(
            image_features[:, 6], image_features[:, 7], image_features[:, 15]
        )
        row3_8_n = self.logical_NOT(row3_8)
        choice8 = self.logical_OR(n_row1_row2_or, row3_8)
        limits.append(row3_8)
        limits.append(row3_8_n)
        limits.append(choice8)
        return choice1, choice2, choice3, choice4, choice5, choice6, choice7, choice8