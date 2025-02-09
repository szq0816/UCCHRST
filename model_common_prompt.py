import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, Linear
from torch.nn.functional import normalize
from torch.nn import functional as F


class FuseTransEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nhead):
        super(FuseTransEncoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model / 2)

        self.common = nn.Linear(self.d_model, self.sigal_d)

        #####
        self.dynamic_scalar = nn.Sequential(nn.Linear(self.sigal_d, self.sigal_d), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(self.sigal_d, 1), nn.Sigmoid())




    def forward(self, tokens):
        encoder_X = self.transformerEncoder(tokens)
        encoder_X_r = encoder_X.reshape(-1, self.d_model)
        encoder_X_r = normalize(encoder_X_r, p=2, dim=1)
        # img, txt = encoder_X_r[:, :self.sigal_d], encoder_X_r[:, self.sigal_d:]
        encoder_X_r = self.common(encoder_X_r)

        dynamic_scalar = self.dynamic_scalar(encoder_X_r)
        # output_img = encoder_X_r + dynamic_scalar * rawdata[:, :self.sigal_d]
        # output_txt = encoder_X_r + dynamic_scalar * rawdata[:, self.sigal_d:]

        return encoder_X_r, dynamic_scalar




class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()

    def _ff_block(self, x):
        x = normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output


class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024, 128, 1024], dropout=0.1):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()

    def _ff_block(self, x):
        x = normalize(x, p=2, dim=1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out

    def forward(self, X):
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p=2, dim=1)
        return mlp_output


class Combiner(nn.Module):
    """
    Combiner module which once trained fuses textual and visual information
    """

    def __init__(self, blip_feature_dim: int, projection_dim: int, hidden_dim: int):
        """
        :param blip_feature_dim: BLIP input feature dimension
        :param projection_dim: projection dimension
        :param hidden_dim: hidden dimension
        """
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(blip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(blip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, blip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    def forward(self, image_features: torch.tensor, text_features: torch.tensor,
                target_features: torch.tensor) -> torch.tensor:
        """
        Takes as input a triplet: image_features, text_features and target_features and outputs the logits which are
        the normalized dot product between the predicted features and the target_features.
        The logits are also multiplied by logit_scale parameter
        :param image_features: BLIP reference image features
        :param text_features: BLIP relative caption features
        :param target_features: BLIP target image features
        :return: scaled logits
        """
        predicted_features = self.combine_features(image_features, text_features)  # * [bsz, dim]

        target_features = self.project_targets(target_features)
        target_features = F.normalize(target_features, dim=-1)  # * [bsz, dim]

        logits = self.logit_scale * predicted_features @ target_features.T  # * [bsz, bsz]
        return logits, predicted_features, target_features

    def project_targets(self, target_features: torch.tensor) -> torch.tensor:
        return target_features

    def combine_features(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        """
        Combine the reference image features and the caption features. It outputs the predicted features
        :param image_features: BLIP reference image features
        :param text_features: BLIP relative caption features
        :return: predicted features
        """
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        output = self.output_layer(combined_features) + dynamic_scalar * text_features + (
                1 - dynamic_scalar) * image_features
        return F.normalize(output, dim=-1)


if __name__ == '__main__':
    combine = Combiner(blip_feature_dim=16, projection_dim=16, hidden_dim=16)


    # a = combine(image_features, text_features, target_features)
