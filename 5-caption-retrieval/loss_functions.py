"""
Loss functions for EEG-to-CLIP alignment.

Implements various loss functions for training EEG encoders to align with CLIP embeddings:
- InfoNCE (naive contrastive learning)
- Knowledge Distillation (similarity-based and logit-based)
- Debiased/Soft Negative Contrastive Loss
- Category-Level Cross-Entropy Supervision
- Combined objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE loss (bidirectional).
    
    Treats each EEG-caption pair as positive, all others in batch as negatives.
    Computes loss in both directions: EEG->Text and Text->EEG.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_eeg: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_eeg: [B, d] normalized EEG embeddings
            z_txt: [B, d] normalized text embeddings
        
        Returns:
            loss: scalar
        """
        # Similarity matrix [B, B]
        logits = (z_eeg @ z_txt.T) / self.temperature
        
        # Targets are diagonals (i matches i)
        targets = torch.arange(z_eeg.size(0), device=z_eeg.device)
        
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        
        loss = 0.5 * (loss_i2t + loss_t2i)
        return loss


class SimilarityBasedKDLoss(nn.Module):
    """
    Similarity-based Knowledge Distillation Loss.
    
    Aligns EEG embeddings directly with text embeddings via cosine similarity.
    L = 1 - z_eeg^T · z_txt
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, z_eeg: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_eeg: [B, d] normalized EEG embeddings
            z_txt: [B, d] normalized text embeddings
        
        Returns:
            loss: scalar (average cosine distance over batch)
        """
        # Cosine similarity: z_eeg^T · z_txt for diagonal elements
        cosine_sims = (z_eeg * z_txt).sum(dim=1)  # [B]
        
        # Loss: 1 - cosine_sim
        loss = (1.0 - cosine_sims).mean()
        return loss


class LogitBasedKDLoss(nn.Module):
    """
    Logit-based Knowledge Distillation Loss using KL divergence.
    
    Student EEG distribution is trained to match teacher CLIP distribution
    over all captions in the embedding bank.
    
    Uses precomputed teacher logits from CLIP's image-to-caption similarities.
    """
    def __init__(self, temperature_student: float = 0.07, temperature_teacher: float = 0.01):
        super().__init__()
        self.temp_student = temperature_student
        self.temp_teacher = temperature_teacher
    
    def forward(
        self,
        z_eeg: torch.Tensor,
        z_txt_batch: torch.Tensor,
        text_emb_bank: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_eeg: [B, d] normalized EEG embeddings
            z_txt_batch: [B, d] normalized text embeddings (matched captions)
            text_emb_bank: [N_caps, d] all caption embeddings (frozen)
            teacher_logits: [B, N_caps] precomputed CLIP image-to-caption logits
        
        Returns:
            loss: scalar
        """
        # Student logits: EEG to all captions
        student_logits = z_eeg @ text_emb_bank.T  # [B, N_caps]
        student_logits = student_logits / self.temp_student
        
        # Teacher logits already scaled by teacher temperature
        teacher_probs = F.softmax(teacher_logits / self.temp_teacher, dim=1)
        
        # Student log probabilities
        student_log_probs = F.log_softmax(student_logits, dim=1)
        
        # KL divergence: KL(teacher || student)
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        return loss


class DebiasedContrastiveLoss(nn.Module):
    """
    Debiased/Soft Negative Contrastive Loss.
    
    Reduces penalty for semantically similar negative captions by:
    - Down-weighting negatives with high similarity to the positive
    - Removes top-k similar negatives from penalty computation
    
    Uses caption embeddings to determine semantic similarity.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        strategy: str = "downweight",  # "downweight" or "removal"
        similarity_threshold: float = 0.8,
        top_k_removal: int = 5,
    ):
        super().__init__()
        self.temperature = temperature
        self.strategy = strategy
        self.similarity_threshold = similarity_threshold
        self.top_k_removal = top_k_removal
    
    def forward(
        self,
        z_eeg: torch.Tensor,
        z_txt: torch.Tensor,
        text_emb_bank: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_eeg: [B, d] normalized EEG embeddings
            z_txt: [B, d] normalized text embeddings (matched captions)
            text_emb_bank: [N_caps, d] all caption embeddings
        
        Returns:
            loss: scalar
        """
        B = z_eeg.size(0)
        device = z_eeg.device
        
        # Similarity matrix [B, N_caps]
        logits = z_eeg @ text_emb_bank.T
        
        if self.strategy == "downweight":
            # Compute semantic similarity between matched captions and all captions
            # z_txt: [B, d], text_emb_bank: [N_caps, d]
            caption_sims = z_txt @ text_emb_bank.T  # [B, N_caps]
            
            # Create weights: high sim -> low weight
            weights = torch.ones_like(logits)
            weights = weights * (1.0 - caption_sims.clamp(0, 1))
            weights = weights.clamp(min=0.1)  # minimum weight to avoid numerical issues
            
            # Scale logits
            logits_scaled = logits / self.temperature
            
            # Weighted cross-entropy
            targets = torch.arange(B, device=device)
            
            # Manual weighted cross-entropy
            log_probs = F.log_softmax(logits_scaled, dim=1)
            weighted_log_probs = log_probs * weights
            loss = -weighted_log_probs.gather(1, targets.unsqueeze(1)).mean()
            
            return loss
        
        elif self.strategy == "removal":
            # Remove top-k most similar negatives from loss computation
            caption_sims = z_txt @ text_emb_bank.T  # [B, N_caps]
            
            # For each sample, identify top-k similar captions (excluding self)
            # Set their logits to -inf so they don't contribute to loss
            for i in range(B):
                # Get similarities for this sample
                sims = caption_sims[i]
                
                # Find top-k (excluding self at position i)
                top_k_indices = torch.topk(sims, k=self.top_k_removal + 1, dim=0)[1]
                top_k_indices = top_k_indices[top_k_indices != i][:self.top_k_removal]
                
                # Mask out top-k from logits
                logits[i, top_k_indices] = float('-inf')
            
            logits_scaled = logits / self.temperature
            targets = torch.arange(B, device=device)
            loss = F.cross_entropy(logits_scaled, targets)
            
            return loss
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


class CategoryCrossEntropyLoss(nn.Module):
    """
    Category-level supervision loss.
    
    Adds a classification head on EEG features to predict object categories.
    Encourages EEG embeddings to capture category-level structure.
    """
    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, class_logits: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            class_logits: [B, num_classes] logits from classification head
            class_labels: [B] class indices
        
        Returns:
            loss: scalar
        """
        loss = self.loss_fn(class_logits, class_labels)
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss objective with flexible weighting.
    
    L_total = λ_infonce·L_infonce + λ_kd_sim·L_kd_sim + λ_debiased·L_debiased + λ_ce·L_ce
    """
    def __init__(
        self,
        lambda_infonce: float = 1.0,
        lambda_kd_sim: float = 0.0,
        lambda_kd_logit: float = 0.0,
        lambda_debiased: float = 0.0,
        lambda_ce: float = 0.0,
        temperature: float = 0.07,
        num_classes: int = 20,
    ):
        super().__init__()
        self.lambda_infonce = lambda_infonce
        self.lambda_kd_sim = lambda_kd_sim
        self.lambda_kd_logit = lambda_kd_logit
        self.lambda_debiased = lambda_debiased
        self.lambda_ce = lambda_ce
        
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.kd_sim_loss = SimilarityBasedKDLoss()
        self.kd_logit_loss = LogitBasedKDLoss()
        self.debiased_loss = DebiasedContrastiveLoss(
            temperature=temperature,
            strategy="downweight",
        )
        self.ce_loss = CategoryCrossEntropyLoss(num_classes=num_classes)
    
    def forward(
        self,
        z_eeg: torch.Tensor,
        z_txt: torch.Tensor,
        class_logits: torch.Tensor = None,
        class_labels: torch.Tensor = None,
        text_emb_bank: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            z_eeg: [B, d] normalized EEG embeddings
            z_txt: [B, d] normalized text embeddings (matched captions)
            class_logits: [B, num_classes] classification logits (optional)
            class_labels: [B] class labels (optional)
            text_emb_bank: [N_caps, d] all caption embeddings (optional, for debiased/kd_logit)
            teacher_logits: [B, N_caps] teacher logits (optional, for kd_logit)
        
        Returns:
            loss: scalar
        """
        total_loss = 0.0
        loss_dict = {}
        
        if self.lambda_infonce > 0:
            loss_infonce = self.infonce_loss(z_eeg, z_txt)
            total_loss += self.lambda_infonce * loss_infonce
            loss_dict['infonce'] = loss_infonce.item()
        
        if self.lambda_kd_sim > 0:
            loss_kd_sim = self.kd_sim_loss(z_eeg, z_txt)
            total_loss += self.lambda_kd_sim * loss_kd_sim
            loss_dict['kd_sim'] = loss_kd_sim.item()
        
        if self.lambda_kd_logit > 0 and text_emb_bank is not None and teacher_logits is not None:
            loss_kd_logit = self.kd_logit_loss(z_eeg, z_txt, text_emb_bank, teacher_logits)
            total_loss += self.lambda_kd_logit * loss_kd_logit
            loss_dict['kd_logit'] = loss_kd_logit.item()
        
        if self.lambda_debiased > 0 and text_emb_bank is not None:
            loss_debiased = self.debiased_loss(z_eeg, z_txt, text_emb_bank)
            total_loss += self.lambda_debiased * loss_debiased
            loss_dict['debiased'] = loss_debiased.item()
        
        if self.lambda_ce > 0 and class_logits is not None and class_labels is not None:
            loss_ce = self.ce_loss(class_logits, class_labels)
            total_loss += self.lambda_ce * loss_ce
            loss_dict['ce'] = loss_ce.item()
        
        return total_loss, loss_dict


def get_loss_fn(loss_config: dict, device: torch.device) -> nn.Module:
    """
    Factory function to instantiate loss function based on config.
    
    Args:
        loss_config: dict with keys:
            - 'type': loss function type (see Loss_TYPES)
            - additional hyperparameters specific to the loss type
        device: torch device
    
    Returns:
        loss_module: instantiated loss module
    """
    loss_type = loss_config.get('type', 'infonce').lower()
    
    if loss_type == 'infonce':
        return InfoNCELoss(
            temperature=loss_config.get('temperature', 0.07)
        ).to(device)
    
    elif loss_type == 'kd_sim':
        return SimilarityBasedKDLoss().to(device)
    
    elif loss_type == 'kd_logit':
        return LogitBasedKDLoss(
            temperature_student=loss_config.get('temperature_student', 0.07),
            temperature_teacher=loss_config.get('temperature_teacher', 0.01),
        ).to(device)
    
    elif loss_type == 'debiased':
        return DebiasedContrastiveLoss(
            temperature=loss_config.get('temperature', 0.07),
            strategy=loss_config.get('strategy', 'downweight'),
            similarity_threshold=loss_config.get('similarity_threshold', 0.8),
            top_k_removal=loss_config.get('top_k_removal', 5),
        ).to(device)
    
    elif loss_type == 'combined':
        return CombinedLoss(
            lambda_infonce=loss_config.get('lambda_infonce', 1.0),
            lambda_kd_sim=loss_config.get('lambda_kd_sim', 0.0),
            lambda_kd_logit=loss_config.get('lambda_kd_logit', 0.0),
            lambda_debiased=loss_config.get('lambda_debiased', 0.0),
            lambda_ce=loss_config.get('lambda_ce', 0.0),
            temperature=loss_config.get('temperature', 0.07),
            num_classes=loss_config.get('num_classes', 20),
        ).to(device)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Available loss functions
LOSS_TYPES = {
    'infonce': 'Standard InfoNCE bidirectional contrastive loss',
    'kd_sim': 'Similarity-based knowledge distillation (cosine alignment)',
    'kd_logit': 'Logit-based knowledge distillation (KL divergence)',
    'debiased': 'Debiased contrastive loss with soft negatives',
    'combined': 'Combined objective with weighted loss components',
}
