import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel


# ==============================================================================
# BLOCK 1: SPARSEMAX
# Thay Softmax — ép trọng số vùng ảnh/từ rác về đúng 0 (sparse attention)
# Có thể visualize rõ ràng hơn Softmax → tốt cho phần Interpretability trong paper
# ==============================================================================
class Sparsemax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        original_size = x.size()
        x = x.view(-1, x.size(self.dim))

        # Numerical stability
        x = x - torch.max(x, dim=1, keepdim=True)[0]

        zs = torch.sort(x, dim=1, descending=True)[0]
        k  = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
        bound = 1 + k * zs
        cumsum = torch.cumsum(zs, dim=1)
        is_gt  = (bound > cumsum).float()
        k_max  = (is_gt * k).max(dim=1, keepdim=True)[0]
        tau    = (is_gt * zs).sum(dim=1, keepdim=True).sub(1).div(k_max)

        return torch.clamp(x - tau, min=0.0).view(original_size)


# ==============================================================================
# BLOCK 2: METADATA-GUIDED SPARSE ATTENTION
# Novelty chính: metadata điều chỉnh Query TRƯỚC khi tính attention score
# → text "nhìn" ảnh qua lăng kính của nguồn báo và chuyên mục
# Tên đặt rõ ràng, không overclaim "Causal" nếu chưa implement backdoor adjustment
# ==============================================================================
class MetadataGuidedSparseAttention(nn.Module):
    def __init__(self, text_dim, img_dim, meta_dim, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(text_dim, hidden_dim)
        self.k_proj = nn.Linear(img_dim,  hidden_dim)
        self.v_proj = nn.Linear(img_dim,  hidden_dim)

        # Gate nhân vào Q: metadata điều chế cách text truy vấn ảnh
        self.meta_gate = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.sparsemax = Sparsemax(dim=-1)
        self.scale     = hidden_dim ** -0.5

    def forward(self, text_feats, img_feats, meta_embeds):
        """
        text_feats  : [B, seq_len, text_dim]
        img_feats   : [B, num_patches, img_dim]   num_patches = 49 (7×7)
        meta_embeds : [B, meta_dim]
        """
        Q = self.q_proj(text_feats)   # [B, seq, H]
        K = self.k_proj(img_feats)    # [B, 49,  H]
        V = self.v_proj(img_feats)    # [B, 49,  H]

        # Điều chế Query bằng metadata gate
        gate    = self.meta_gate(meta_embeds).unsqueeze(1)  # [B, 1, H]
        Q_gated = Q * gate                                  # [B, seq, H]

        # Attention scores + Sparsemax
        scores  = torch.bmm(Q_gated, K.transpose(1, 2)) * self.scale  # [B, seq, 49]
        weights = self.sparsemax(scores)                               # sparse

        # Context vector
        context = torch.bmm(weights, V)  # [B, seq, H]

        return context, weights


# ==============================================================================
# BLOCK 3: KIẾN TRÚC TỔNG THỂ
# ==============================================================================
class ClickbaitDetector(nn.Module):
    def __init__(
        self,
        num_sources,
        num_categories,
        phobert_name='vinai/phobert-base',
        cnn_channels=128,
        hidden_dim=256,
        meta_embed_dim=64,
        dropout=0.3,
        freeze_phobert_layers=6,   # Freeze N layer đầu để tiết kiệm VRAM
    ):
        super().__init__()

        # ── NHÁNH VĂN BẢN ────────────────────────────────────────────────────
        self.phobert = AutoModel.from_pretrained(phobert_name)

        # Freeze embeddings (position, token type, word embeddings)
        for param in self.phobert.embeddings.parameters():
            param.requires_grad = False

        # Freeze N layer đầu của encoder
        for i, layer in enumerate(self.phobert.encoder.layer):
            if i < freeze_phobert_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # Freeze pooler
        for param in self.phobert.pooler.parameters():
            param.requires_grad = False

        # Conv1D trích xuất n-gram đặc trưng "giật tít"
        # Kernel 2 và 3 chạy song song → concat → project
        self.conv2 = nn.Sequential(
            nn.Conv1d(768, cnn_channels, kernel_size=2, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(768, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU()
        )
        # Project từ 2*cnn_channels xuống cnn_channels để giữ chiều nhất quán
        self.text_proj = nn.Linear(cnn_channels * 2, cnn_channels)

        # ── NHÁNH HÌNH ẢNH ───────────────────────────────────────────────────
        resnet = models.resnet50(weights=None)  # weights được khôi phục từ checkpoint deploy
        # Bỏ AvgPool và FC cuối → giữ feature map [B, 2048, 7, 7]
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze toàn bộ ResNet — chỉ fine-tune qua bottleneck
        for param in self.resnet_features.parameters():
            param.requires_grad = False

        # Bottleneck 1×1: 2048 → hidden_dim, giảm FLOPs
        self.img_bottleneck = nn.Sequential(
            nn.Conv2d(2048, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )

        # ── NHÁNH METADATA ────────────────────────────────────────────────────
        # Index 0 = UNK, nên num_embeddings = num + 1
        self.src_embed = nn.Embedding(num_sources   + 1, meta_embed_dim, padding_idx=0)
        self.cat_embed = nn.Embedding(num_categories + 1, meta_embed_dim, padding_idx=0)
        meta_dim = meta_embed_dim * 2   # 128

        # ── TẦNG FUSION ──────────────────────────────────────────────────────
        self.fusion = MetadataGuidedSparseAttention(
            text_dim=cnn_channels,
            img_dim=hidden_dim,
            meta_dim=meta_dim,
            hidden_dim=hidden_dim,
        )

        # ── CLASSIFIER ───────────────────────────────────────────────────────
        # Input: pooled_text [cnn_channels] + pooled_context [hidden_dim]
        clf_input_dim = cnn_channels + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)   # raw logit, không Sigmoid (dùng BCEWithLogitsLoss)
        )

    # --------------------------------------------------------------------------
    def forward(self, input_ids, attention_mask, images, source_ids, category_ids):
        # ── A. VĂN BẢN ───────────────────────────────────────────────────────
        bert_out = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state                       # [B, seq, 768]

        x = bert_out.permute(0, 2, 1)            # [B, 768, seq]  (Conv1D cần C trước)

        c2 = self.conv2(x)                        # [B, C, seq+1]  (padding=1, kernel=2)
        c3 = self.conv3(x)                        # [B, C, seq]    (padding=1, kernel=3)

        # Crop về cùng độ dài (seq) rồi concat
        min_len  = min(c2.size(2), c3.size(2))
        c2, c3   = c2[:, :, :min_len], c3[:, :, :min_len]
        text_cat = torch.cat([c2, c3], dim=1)    # [B, 2C, seq]

        text_cat  = text_cat.permute(0, 2, 1)    # [B, seq, 2C]
        text_feats = self.text_proj(text_cat)     # [B, seq, C]

        # ── B. HÌNH ẢNH ──────────────────────────────────────────────────────
        with torch.no_grad():
            img_map = self.resnet_features(images)     # [B, 2048, 7, 7]

        img_map   = self.img_bottleneck(img_map)       # [B, H, 7, 7]
        B, C, H, W = img_map.shape
        img_feats = img_map.view(B, C, H * W).permute(0, 2, 1)  # [B, 49, H]

        # ── C. METADATA ──────────────────────────────────────────────────────
        s_emb       = self.src_embed(source_ids)      # [B, 64]
        c_emb       = self.cat_embed(category_ids)    # [B, 64]
        meta_embeds = torch.cat([s_emb, c_emb], dim=1)  # [B, 128]

        # ── D. FUSION ────────────────────────────────────────────────────────
        context, attn_weights = self.fusion(text_feats, img_feats, meta_embeds)
        # context: [B, seq, H]   attn_weights: [B, seq, 49]

        # ── E. POOLING + CLASSIFY ────────────────────────────────────────────
        mask = attention_mask[:, :text_feats.size(1)].unsqueeze(-1).float()

        pooled_text = (text_feats * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)   # [B, C]
        pooled_context = (context * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)      # [B, H]

        final_repr = torch.cat([pooled_text, pooled_context], dim=1)  # [B, C+H]
        logits     = self.classifier(final_repr).squeeze(-1)           # [B]

        return logits, attn_weights   # attn_weights dùng để visualize sau


# ==============================================================================
# QUICK SANITY CHECK
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ClickbaitDetector(
        num_sources=10,
        num_categories=15,
        phobert_name='vinai/phobert-base',
        cnn_channels=128,
        hidden_dim=256,
        meta_embed_dim=64,
        dropout=0.3,
        freeze_phobert_layers=6,
    ).to(device)

    # Đếm params
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    # Dummy batch
    B = 4
    dummy = {
        "input_ids":      torch.randint(0, 1000, (B, 256)).to(device),
        "attention_mask": torch.ones(B, 256, dtype=torch.long).to(device),
        "images":         torch.randn(B, 3, 224, 224).to(device),
        "source_ids":     torch.randint(0, 10,  (B,)).to(device),
        "category_ids":   torch.randint(0, 15,  (B,)).to(device),
    }

    logits, attn = model(**dummy)
    print(f"Logits shape    : {logits.shape}")    # [4]
    print(f"Attn shape      : {attn.shape}")      # [4, seq, 49]
    print(f"Logits sample   : {logits.detach().cpu()}")