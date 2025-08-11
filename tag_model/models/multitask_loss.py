class MultiTaskTaggingLoss(nn.Module):
    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights
        

        self.global_loss = AsymmetricLoss(gamma_neg=7, gamma_pos=0, clip=0.05)
        self.local_loss = AsymmetricLoss(gamma_neg=5, gamma_pos=1, clip=0.03)
        self.relation_loss = RelationLoss()
        

        self.consistency_loss = nn.MSELoss()
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict containing 'global', 'local', 'relation' logits
            targets: dict containing corresponding ground truth labels
        """
        total_loss = 0
        loss_dict = {}
        

        if 'global' in predictions:
            global_loss = self.global_loss(predictions['global'], targets['global'])
            total_loss += self.loss_weights['global'] * global_loss
            loss_dict['global_loss'] = global_loss
            

        if 'local' in predictions:
            local_loss = self.local_loss(predictions['local'], targets['local'])
            total_loss += self.loss_weights['local'] * local_loss
            loss_dict['local_loss'] = local_loss
            

        if 'relation' in predictions:
            relation_loss = self.relation_loss(predictions['relation'], targets['relation'])
            total_loss += self.loss_weights['relation'] * relation_loss
            loss_dict['relation_loss'] = relation_loss
            
        if 'global' in predictions and 'local' in predictions:
            global_prob = torch.sigmoid(predictions['global'])
            local_prob = torch.sigmoid(predictions['local'])
            
            consistency_loss = self.consistency_loss(
                global_prob, 
                torch.max(local_prob, dim=-1)[0].unsqueeze(-1).expand_as(global_prob)
            )
            total_loss += self.loss_weights['consistency'] * consistency_loss
            loss_dict['consistency_loss'] = consistency_loss
            
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict