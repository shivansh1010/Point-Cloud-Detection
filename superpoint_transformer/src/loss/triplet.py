import torch
import torch.nn as nn

__all__ = ['TripletLoss']

def cudafy(module):
    if torch.cuda.is_available():
        return module.cuda()
    else:
        return module.cpu()


class TripletLoss(nn.Module):
    """
    the same with tf.triplet_semihard_loss
    Shape:
        - Input: :math:`(N, C)` where `C = number of channels`
        - Target: :math:`(N)`
        - Output: scalar.
    """

    def __init__(self,
                 weight=None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        print("reduction = ", reduction)
        print("weight = ", weight)
        super(TripletLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.relu = nn.ReLU()

    def __repr__(self):
        arg_keys = ['weight', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def masked_maximum(self, data, mask, dim=1):
        """Computes the axis wise maximum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the maximum.
            Returns:
              masked_maximums: N-D `Tensor`.
                The maximized dimension is of size 1 after the operation.
            """
        axis_minimums = torch.min(data, dim, keepdim=True).values
        masked_maximums = torch.max(torch.mul(data - axis_minimums, mask), dim, keepdim=True).values + axis_minimums
        return masked_maximums

    def masked_minimum(self, data, mask, dim=1):
        """Computes the axis wise minimum over chosen elements.
            Args:
              data: 2-D float `Tensor` of size [n, m].
              mask: 2-D Boolean `Tensor` of size [n, m].
              dim: The dimension over which to compute the minimum.
            Returns:
              masked_minimums: N-D `Tensor`.
                The minimized dimension is of size 1 after the operation.
            """
        axis_maximums = torch.max(data, dim, keepdim=True).values
        masked_minimums = torch.min(torch.mul(data - axis_maximums, mask), dim, keepdim=True).values + axis_maximums
        return masked_minimums

    def pairwise_distance(self, embeddings, squared=True):
        pairwise_distances_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True) + \
                                     torch.sum(embeddings.t() ** 2, dim=0, keepdim=True) - \
                                     2.0 * torch.matmul(embeddings, embeddings.t())

        error_mask = pairwise_distances_squared <= 0.0
        if squared:
            pairwise_distances = pairwise_distances_squared.clamp(min=0)
        else:
            pairwise_distances = pairwise_distances_squared.clamp(min=1e-16).sqrt()

        pairwise_distances = torch.mul(pairwise_distances, ~error_mask)

        num_data = embeddings.shape[0]
        # Explicitly set diagonals to zero.
        mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(cudafy(torch.ones([num_data])))
        pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)
        return pairwise_distances

    def forward(self, embeddings, target, margin=0.2, squared=True):
        # Split the data into multiple parts and sum them to manage cuda memory
        # We can have max of 16000 points.

        print("Embeddings shape = ", embeddings.shape)
        print("self weight = ", self.weight)
        print("target shape = ", target.shape)
        data_count = 360
        loop_count = int(embeddings.shape[0]/data_count)
        preds = torch.argmax(embeddings, dim=1)
        diff = torch.sum(((target - preds) **2) > 1e-16)
        loss = diff
        #return loss

        # Normalize embeddings...
        copy_embeddings = embeddings
        new_target = torch.stack((target, target, target, target, target, target, target, target, target), -1)
        embeddings = torch.abs(torch.sub(embeddings , new_target))
        #l2norm = torch.sum(embeddings ** 2, dim=1, keepdim=True)
        #embeddings = torch.div(embeddings, l2norm);

        for i in range(loop_count -1):
            loss_metric = online_batch_all(embeddings[i*data_count:(i+1)*data_count], target[i*data_count:(i+1)*data_count], margin, squared, normalize=True, relu=self.relu)
            loss = loss + loss_metric[0]*data_count

        loss_metric = online_batch_all(embeddings[(loop_count -1)*data_count:], target[(loop_count - 1)*data_count:], margin, squared, normalize=True, relu=self.relu)
        data_len = len(target[(loop_count -1)*data_count:])
        loss = loss + loss_metric[0]*data_len
        loss = loss/embeddings.shape[0]
        print("Loss = ", loss)
        return loss

    def forward_sub1(self, embeddings, target, margin=0.2, squared=True):
        """
        :param features: [B * N features]
        :param target: [B]
        :param square: if the distance squared or not.
        :return:
        """
        lshape = target.shape
        assert len(lshape) == 1
        labels = target.int().unsqueeze(-1)  # [B, 1]
        pdist_matrix = self.pairwise_distance(embeddings, squared=squared)

        adjacency = labels == torch.transpose(labels, 0, 1)

        adjacency_not = ~adjacency
        batch_size = labels.shape[0]

        # Compute the mask

        pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])

        mask = adjacency_not.repeat([batch_size, 1]) & (pdist_matrix_tile > torch.reshape(
            torch.transpose(pdist_matrix, 0, 1), [-1, 1]))

        mask_final = torch.reshape(torch.sum(mask.float(), 1, keepdim=True) >
                                   0.0, [batch_size, batch_size])
        mask_final = torch.transpose(mask_final, 0, 1)

        adjacency_not = adjacency_not.float()
        mask = mask.float()

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = torch.reshape(
            self.masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
        negatives_outside = torch.transpose(negatives_outside, 0, 1)

        # negatives_inside: largest D_an.
        negatives_inside = self.masked_maximum(pdist_matrix, adjacency_not).repeat([1, batch_size])
        semi_hard_negatives = torch.where(mask_final, negatives_outside, negatives_inside)

        loss_mat = torch.add(margin, pdist_matrix - semi_hard_negatives)

        mask_positives = adjacency.float() - torch.diag(cudafy(torch.ones([batch_size])))

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        num_positives = torch.sum(mask_positives)

        triplet_loss = torch.div(torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0)), num_positives)

        # triplet_loss = torch.sum(torch.mul(loss_mat, mask_positives).clamp(min=0.0))
        return triplet_loss

"""
    See omoindrot's blog for the really useful breakdown
    and broadcasting techniques
    https://omoindrot.github.io/triplet-loss
"""

def online_batch_all(embeddings, labels, margin=0.5, squared=False, normalize=False, device='cuda', relu=None):
    ''' Returns the triplet loss over a batch of embeddings, given class labels.
        Only 'semi-hard' triplets are counted i.e a_p - a_n + margin > 0
    Args:
        embeddings: tensor of shape (batch_size, embedding_dim)
        labels: integer labels of the batch, of size (batch_size, )
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
        num_valid_triplets: total number of mined triplets
        fraction_positive_triplets: fraction of valid triplets used to compute the loss
    '''
    # pairwise embedding distances
    p_dist = _pairwise_distances(embeddings, squared=squared, relu=relu)

    # anchor to positive (batch_size, batch_size, 1)
    a_p = p_dist.unsqueeze(2)
    # anchor to positive (batch_size, 1, batch_size)
    a_n = p_dist.unsqueeze(1)

    # mask of valid triplets (batch_size, batch_size, batch_size
    # True if [i, j, k] is a valid (a, p, n) triplet
    valid_triplet_mask = _triplet_mask_all(labels, device)

    # create triplet tensor (batch_size, batch_size, batch_size)
    triplet_loss = a_p - a_n + margin

    # zero the non-valid triplets
    triplet_loss = triplet_loss * valid_triplet_mask.float()

    # Remove non-semi-hard triplets (easy)
    # i.e when a_p + margin < a_n
    triplet_loss = torch.max(triplet_loss, torch.Tensor([0.0]).to(device))

    # Get the number of triplets greater than 0
    valid_triplets = (triplet_loss > 1e-16).float()

    num_positive_triplets = torch.sum(valid_triplets)

    num_valid_triplets = torch.sum(valid_triplet_mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, num_positive_triplets, num_valid_triplets

def _pairwise_distances(embeddings, squared=True, relu=None):
    ''' Returns pairwise distances for a batch of embeddings
        Computational reference:
        https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf
    Args:
        embeddings: tensor of shape (batch_size, embedding_dim)
        squared: the squared euclidean distance matrix is computed when true
    Returns:
        pairwise distances between all the embeddings of shape (batch_size, batch_size)
    '''

    gram_matrix = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))

    diag = torch.diag(gram_matrix)

    # D(x, y) = ||x||^2 - 2 <a, b> + ||y||^2
    dists = diag + diag.T - 2 * gram_matrix

    #print("dists = ", dists)
    if not squared and relu is not None:
        # sqrt produces zero values for infinite gradiences
        # add double precision epsilon
        dists = torch.sqrt(dists + 1e-16)

        # clamp negative values that occur due to lack of floating point precision
        dists = relu(dists)

    return dists

def _triplet_mask_all(labels, device):
    '''Returns the 3-D mask [a, p, n] where True is a valid a-p-n triplet
    Args:
        labels: (batch_size, )
    Returns:
        triplet_mask: (batch_size, batch_size, batch_size)

    '''
    # create 3-D tensor [a, p, n] where labels[a] != labels[p] != labels[n]
    # get anchors
    positive_labels = labels.unsqueeze(1) == labels
    # get the anchor idxs i = j
    idxs_not_anchor = (torch.eye(labels.shape[0]) == 0).to(device)
    # combine anchors with positives [a, p, 1]
    anchor_positive = (positive_labels & idxs_not_anchor).unsqueeze(2)
    # get the negative labels [a, 1, n]
    anchor_negative = ~positive_labels.unsqueeze(1)

    # Tensor of the valid triplets [i, j, k] True, if
    triplet_mask = (anchor_positive & anchor_negative).to(device)

    # mask = idxs & valid_triplets

    return triplet_mask

def _online_hard(labels):
    # anchor_positive_mask = _get_anchor_positive_mask(labels)
    # anchor_negative_mask = _get_anchor_negative_mask(labels)
    return NotImplemented

def _get_anchor_positive_mask(labels):
    '''Returns the 2-D mask [a, p] where 1 is a valid a-p pair
    Args:
        labels: (batch_size, )
    Returns:
        anchor_positive_mask: (batch_size, batch_size)

    '''
    # find equal label idxs
    # broadcast label tensor and perform outer product
    positive_idxs = labels.unsqueeze(1) == labels

    idxs_not_anchor = torch.eye(labels.shape[0]) == 0

    anchor_positive_mask = positive_idxs & idxs_not_anchor

    return anchor_positive_mask

def _get_anchor_negative_mask(labels):
    positive_idxs = labels.unsqueeze(1) == labels
    negative_mask = ~positive_idxs
    return negative_mask