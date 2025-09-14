import torch

class projector:
    def __init__(self, rank, proj_type='std'):
        self.rank = rank
        self.proj_type = proj_type
        self.ortho_matrix = None
        self.st_step_size = -10000
        self.update_rank = 1

    def project(self, full_rank_grad):
        if self.proj_type == 'right':
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)

        return full_rank_grad


    # svd decomposition
    def get_orthogonal_matrix_svd(self, grad, svd_lowrank=False):
        matrix = grad.data.clone()

        if matrix.dtype != torch.float: #torch.linalg.svd doesn't support half precision types such as torch.bfloat16
            float_data = False
            original_type = matrix.dtype
            original_device = matrix.device
            matrix = matrix.float()
        else :
            float_data = True

        if svd_lowrank :
            U, s, V = torch.svd_lowrank(matrix, q = self.rank+2, niter=1) #q a slightly overestimated rank of A
            Vh = V.t()
        else :
            U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)

        if self.proj_type =='right' :
            ortho_matrix = Vh[:self.rank, :]
        elif self.proj_type=='left' :
            ortho_matrix = U[:, :self.rank]

        if not(float_data) :
            ortho_matrix = ortho_matrix.to(device=original_device, dtype=original_type)

        self.ortho_matrix = ortho_matrix


    # def power_iteration(self, matrix, init, intermediate_orthogonalization=False):
    #     if self.proj_type == 'right':
    #         U = matrix @ init.t()
    #         if intermediate_orthogonalization : # Not necessary for computing right singular vectors
    #             U = Gram_Schmidt(U)
    #         projection_map = matrix.t() @ U
    #         del U
    #
    #         projection_map = Gram_Schmidt(projection_map)
    #         self.ortho_matrix = projection_map.t()
    #
    #     elif self.proj_type == 'left':
    #         V = matrix.t() @ init
    #         if intermediate_orthogonalization : # Not necessary for computing left singular vectors
    #             V = Gram_Schmidt(V)
    #         projection_map = matrix @ V
    #         del V
    #
    #         projection_map = Gram_Schmidt(projection_map)
    #         self.ortho_matrix = projection_map

    def power_iteration(self, full_rank_grad, init, intermediate_orthogonalization=False):
        if self.ortho_matrix.dtype != torch.float:
            float_data = False
            original_type = self.ortho_matrix.dtype
            self.ortho_matrix = self.ortho_matrix.float()
        else:
            float_data = True

        random_vector = torch.randn_like(self.ortho_matrix)
        tangent_vector = random_vector

        U, Sigma, V = self.rank_k_matrix_estimation(tangent_vector, k=self.update_rank)

        self.ortho_matrix = torch.matmul(
            torch.matmul(
                torch.concat([torch.matmul(self.ortho_matrix, V), U], 1),
                torch.concat([torch.cos(self.st_step_size * Sigma), torch.sin(self.st_step_size * Sigma)], 0)
            ).reshape((self.ortho_matrix.shape[0]), Sigma.shape[0]), V.t()
        ) + torch.matmul(
            self.ortho_matrix, (torch.eye(V.shape[0]).to("cuda") - torch.matmul(V, V.t()))
        )

        if not float_data:
            self.ortho_matrix = self.ortho_matrix.to(original_type)

    def rank_k_matrix_estimation(self, matrix, k=1):

        U, Sigma, Vt = torch.linalg.svd(matrix, full_matrices=False)

        if k == 1:
            return U[:, :k], Sigma[:k], Vt.t()[:, :k]
        return U[:, :k], torch.diag(Sigma[:k]), Vt.t()[:, :k]


def Gram_Schmidt(matrix):
    original_type = matrix.dtype #torch.linalg.qr doesn't support helf precision types such as torch.bfloat16
    matrix, _ = torch.linalg.qr(matrix.to(dtype=torch.float32))
    matrix = matrix.to(dtype=original_type)

    return matrix