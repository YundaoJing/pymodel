import matplotlib.figure
import numpy
import torch
import matplotlib.axes
import matplotlib.pyplot

"""

    :num_input: int
        number of input dimension.
    :num_output: int
        number of output dimension.
    :num_size: int
        number of b-spline channels.
        num_size = num_input * num_output
        (num_output, num_input) <-> (num_size, )
    :num_grid: int
        number of b-spline grid points i.e. knots.
    :num_grid_ext: int
        the grid points interval will be extended in both sides of grid_mat.
    :num_basefunc: int
        number of base function in b-spline.
        num_basefunc = num_grid + 2*num_grid_ext - num_deg_k - 1
    :num_sample_dis: int
        number of b-spline sample points for display/update.
    :num_deg_k: int
        degree of b-spline.

    :mat_input: (num_batch, num_input) torch.Tensor
        matrix of forward input.
    :mat_output: (num_batch, num_output) torch.Tensor
        matrix of forward output.

    :vec_act: (num_size, ) torch.Tensor
        vector of edge activate intensity.
    :vec_mask: (num_size, ) torch.Tensor
        vector of edge mask, i.e enable or not.

    :mat_coef: (num_size, num_basefunc) torch.Tensor/torch.nn.Parameter
        matrix of b-spline base functions coefficient.
    :vec_scale_base: (num_size, ) torch.Tensor/torch.nn.Parameter
        vector of base part scale 
    :vec_scale_bspline: (num_size, ) torch.Tensor/torch.nn.Parameter
        vector of b-spline part scale 

    :mat_range_grid: (num_size, 2) torch.Tensor
        matrix of b-spline grid range.
    :mat_grid: (num_size, num_grid) torch.Tensor
        matrix of the b-spline grid points.

    :mat_basefunc: (num_size, num_basefunc, num_batch) torch.Tensor
        matrix of b-spline base function.
    :mat_x (num_size, num_batch) torch.Tensor
        matrix of b-spline/base input.
    :mat_y: (num_size, num_batch) torch.Tensor
        matrix of b-spline/base output.
    :mat_y_base: (num_size, num_batch) torch.Tensor
        matrix of base output.
    :mat_y_bspline: (num_size, num_batch) torch.Tensor
        matrix of b-spline output.

    :mat_basefunc_dis: (num_size, num_basefunc, num_sample_dis) torch.Tensor
        matrix of b-spline base function for display/update.
    :mat_x_dis (num_size, num_sample_dis) torch.Tensor
        matrix of b-spline/base input for display/update.
    :mat_y_dis: (num_size, num_sample_dis) torch.Tensor
        matrix of b-spline/base output for display/update.
    :mat_y_base_dis: (num_size, num_sample_dis) torch.Tensor
        matrix of base output for display/update.
    :mat_y_bspline_dis: (num_size, num_sample_dis) torch.Tensor
        matrix of b-spline output for display/update.

    :vec_bias: (num_edge, ) torch.Tensor
        vector of edge(KANLayers) bias.
"""

def get_mat_grid(
        mat_range_grid:torch.Tensor,
        num_grid:int,
        num_size:int, 
        device:str='cpu', 
        ) -> torch.Tensor:
    """
    Get b-spline grid matrix

    """
    # (num_grid, )
    mat_grid = torch.linspace(0, 1, num_grid, device=device)
    # (num_grid, ) -> (num_size, num_grid)
    mat_grid = torch.einsum('i,j->ji', mat_grid, torch.ones(num_size, device=device))
    mat_grid = mat_grid * (mat_range_grid[:, [1]] - mat_range_grid[:, [0]]) + mat_range_grid[:, [0]]

    return mat_grid


def get_mat_basefunc(
        mat_x:torch.Tensor,
        mat_grid:torch.Tensor,
        num_deg_k:int=3,
        num_grid_ext:int=2, 
        ) -> torch.Tensor:
    """
    Get b-spline base functions on its sample ponits in different channels.

    """
    for ii in range(num_grid_ext):
        vec_grid_interval = (mat_grid[:, [-1]] - mat_grid[:, [0]]) / (mat_grid.shape[1] - 1)
        mat_grid = torch.cat((mat_grid[:, [0]] - vec_grid_interval, mat_grid), dim=1)
        mat_grid = torch.cat((mat_grid, mat_grid[:, [-1]] + vec_grid_interval), dim=1)

    # (num_size, num_sample) -> (num_size, 1, num_sample)
    mat_x = mat_x.unsqueeze(dim=1)
    # (num_size, num_grid) -> (num_size, num_grid, 1)
    mat_grid = mat_grid.unsqueeze(dim=2)

    if num_deg_k == 0:
        # (num_size, num_grid, num_sample)
        mat_basefunc = (mat_x >= mat_grid[:, :-1]) * (mat_x < mat_grid[:, 1:]) * 1
    else:
        mat_basefunc_km1 = get_mat_basefunc(
            mat_x=mat_x.squeeze(dim=1), 
            mat_grid=mat_grid.squeeze(dim=2), 
            num_deg_k=num_deg_k-1, 
            num_grid_ext=0)  # num_grid_extend must be 0 in inner recursion
        # (num_size, num_basefunc, num_sample)
        mat_weight_1 = (mat_x - mat_grid[:, :-(1+num_deg_k)]) / (mat_grid[:, num_deg_k:-1] - mat_grid[:, :-(1+num_deg_k)])
        mat_weight_2 = (mat_grid[:, num_deg_k+1:] - mat_x) / (mat_grid[:, num_deg_k+1:] - mat_grid[:, 1:-num_deg_k])
        mat_basefunc = mat_weight_1 * mat_basefunc_km1[:, :-1]  + mat_weight_2 * mat_basefunc_km1[:, 1:]

    return mat_basefunc


def get_mat_coef(
        mat_basefunc:torch.Tensor,
        mat_y_bspline:torch.Tensor, 
        method:str='lstsq', 
        device:str='cpu') -> torch.Tensor:
    """
    Get b-spline base functions coefficient by least square method.
    A x X = B
    mat_basefunc.permute(0, 2, 1) * mat_coef[:, :, None] = mat_y_bspline.unsqueeze(dim=2)
    (num_size, num_sample, num_basefunc) * (num_size, num_basefunc, 1) = (num_size, num_sample, 1)

    """
    if method == 'lstsq':

        mat_coef = torch.linalg.lstsq(
            input=mat_basefunc.permute(0, 2, 1),
            b=mat_y_bspline.unsqueeze(dim=2)).solution
        # (num_size, num_basefunc, 1) -> (num_size, num_basefunc)
        mat_coef = mat_coef.squeeze(dim=2)

    elif method == 'iteration':
    
        class Model(torch.nn.Module):

            def __init__(self, mat_coef):

                super().__init__()
                self.mat_coef = torch.nn.Parameter(mat_coef)

            def forward(self, mat_basefunc):

                mat_y_bspline = mat_basefunc.permute(0, 2, 1) @ self.mat_coef[:, :, None]
                mat_y_bspline = mat_y_bspline.squeeze(dim=2)

                return mat_y_bspline

            def train(self, mat_y_bspline):

                epoch = 1000
                learn_rate = 0.03
                criterion = torch.nn.MSELoss(reduction='mean')
                optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)
                for ii in range(epoch):
                    optimizer.zero_grad()
                    mat_y_bspline_pred = self.forward(mat_basefunc=mat_basefunc)
                    loss = criterion(mat_y_bspline_pred, mat_y_bspline)
                    loss.backward()
                    optimizer.step()

        num_size, num_basefunc, _ = mat_basefunc.shape
        model = Model(mat_coef=0.5*(2*torch.rand(num_size, num_basefunc, device=device)-1))
        model.train(mat_y_bspline=mat_y_bspline)
        mat_coef = model.mat_coef.detach()

    return mat_coef


def get_mat_y_bspline(
        mat_basefunc: torch.Tensor,
        mat_coef: torch.Tensor) -> torch.Tensor :
    """
    Get b-spline base functions on its sample ponits in different channels.
    A x X = B
    mat_basefunc.permute(0, 2, 1) * mat_coef[:, :, 0] = value_mat.unsqueeze(dim=2)
    (num_size, num_sample, num_basefunc) * (num_size, num_basefunc, 1) = (num_size, num_sample, 1)

    """
    mat_y_bspline = torch.einsum('ijk,ik->ij', mat_basefunc.permute(0, 2, 1), mat_coef)

    return mat_y_bspline


lib_symbol = {

    '0*(x)': lambda x: x*0,
    '(x)': lambda x: x,
    '(x)^2': lambda x: x**2,
    '(x)^3': lambda x: x**3,
    '(x)^4': lambda x: x**4,
    '1/(x)': lambda x: 1/x,
    '1/(x)^2': lambda x: 1/x**2,
    '1/(x)^3': lambda x: 1/x**3,
    '1/(x)^4': lambda x: 1/x**4,
    'sqrt(x)': lambda x: torch.sqrt(x),
    '1/sqrt(x)': lambda x: 1 / torch.sqrt(x),
    'exp(x)': lambda x: torch.exp(x),
    'ln(x)': lambda x: torch.log(x),
    'abs(x)': lambda x: torch.abs(x),
    'sigmoid(x)': lambda x: torch.sigmoid(x),
    'sign(x)': lambda x: torch.sign(x),
    'sin(x)': lambda x: torch.sin(x),
    'cos(x)': lambda x: torch.cos(x),
    'tan(x)': lambda x: torch.tan(x),
    'sinh(x)': lambda x: torch.sinh(x),
    'cosh(x)': lambda x: torch.cosh(x),
    'tanh(x)': lambda x: torch.tanh(x),
    'arcsin(x)': lambda x: torch.arcsin(x),
    'arccos(x)': lambda x: torch.arccos(x),
    'arctan(x)': lambda x: torch.arctan(x),
    'gaussian(x)': lambda x: torch.exp(-torch.pow(x, 2)), }


def get_vec_y_symbol(
        name_func:str, 
        vec_para:torch.Tensor, 
        vec_x:torch.Tensor, ) -> torch.Tensor:
    
    func = lib_symbol[name_func]
    vec_y = vec_para[2] * func(vec_para[0] * vec_x + vec_para[1]) + vec_para[3]

    return vec_y


def get_vec_para(
        name_func:str,
        vec_x:torch.Tensor,
        vec_y:torch.Tensor,
        list_vec_para_0:list=[1.0, 0.05, 1.0, 0.05]) \
        -> tuple[torch.Tensor, float, torch.nn.Module] :
    """
    Get output y from input x by function (lambda)
    c * func(a * x + b) + d
    vec_para[2] * func(vec_para[0] * x + vec_para[1]) + vec_para[4]

    :func: function
        lambda functions in lib_symbol
    :vec_x: (num_sample, ) torch.Tensor
        input data
    :vec_y: (num_sample, ) torch.Tensor
        output data
    :vec_para_0: (4, ) torch.Tensor
        initial vec_para, i.e. [a, b, c, d]

    :vec_para: (4) torch.Tensor
        optimized vec_para value
    :r2: float
        R^2 of optimization
    :model: torch.nn.Module
        Adam methode to find vec_para, most for debug usage
    """
    class func_model(torch.nn.Module):

        def __init__(self, name_func, vec_para_0):

            super().__init__()
            self.name_func = name_func
            self.vec_para = torch.nn.Parameter(vec_para_0)

        def forward(self, vec_x):

            vec_y = get_vec_y_symbol(
                name_func=self.name_func, 
                vec_para=self.vec_para, vec_x=vec_x)
            
            return vec_y


    vec_para_0 = torch.tensor(list_vec_para_0).to(float)

    epoch = 500
    model = func_model(name_func=name_func, vec_para_0=vec_para_0)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for ii in range(epoch):
        optimizer.zero_grad()
        loss = criterion(model.forward(vec_x), vec_y)
        loss.backward(retain_graph=True)
        optimizer.step()
    r2 = 1 - loss / (torch.var(vec_y) + 1e-8)

    vec_para = model.vec_para.data
    r2 = r2.item()

    return vec_para, r2, model


class KAN_EDGE(torch.nn.Module):

    def __init__(
            self,
            num_input:int,
            num_output:int,
            num_grid:int=10,
            num_grid_ext:int=2,
            num_sample_dis:int=200,
            num_deg_k:int=3,
            range_grid_0:list=[-1, 1],
            scale_base_0:float=1.0,
            scale_bspline_0:float=0.1, 
            device:str='cpu'):

        super().__init__()

        self.device = device

        self.num_input = num_input
        self.num_output = num_output
        self.num_size = num_input * num_output
        self.num_sample_dis = num_sample_dis

        self.num_deg_k = num_deg_k
        self.num_grid = num_grid
        self.num_grid_ext = num_grid_ext
        self.num_basefunc = num_grid + 2*num_grid_ext - num_deg_k - 1
        self.mat_range_grid = torch.einsum(
            'i,j->ij', 
            torch.ones(self.num_size, device=self.device), torch.tensor(range_grid_0, device=self.device))
        self.mat_grid = get_mat_grid(
            device=self.device, 
            mat_range_grid=self.mat_range_grid, num_grid=self.num_grid, num_size=self.num_size, )
        
        # bspline
        self.flag_enable_bspline = True
        self.vec_scale_base = torch.nn.Parameter(
            torch.ones(self.num_size, device=self.device) * scale_base_0)
        self.vec_scale_bspline = torch.nn.Parameter(
            torch.ones(self.num_size, device=self.device) * scale_bspline_0)

        self.vec_act = torch.ones(self.num_size, device=self.device)
        self.mat_x_dis = get_mat_grid(
            device=self.device, 
            mat_range_grid=self.mat_range_grid, num_grid=self.num_sample_dis, num_size=self.num_size)
        self.mat_basefunc_dis = get_mat_basefunc(
            mat_x=self.mat_x_dis, mat_grid=self.mat_grid, num_deg_k=self.num_deg_k, num_grid_ext=self.num_grid_ext)
        self.mat_y_bspline_dis = 2*torch.rand(size=(self.num_size, self.num_sample_dis), device=self.device) - 1
        self.mat_y_base_dis = 2*torch.rand(size=(self.num_size, self.num_sample_dis), device=self.device) - 1
        self.mat_coef = torch.nn.Parameter(
            get_mat_coef(
                device=self.device, 
                mat_basefunc=self.mat_basefunc_dis, mat_y_bspline=self.mat_y_bspline_dis))
        self.mat_y_dis = (
            self.vec_scale_base[:, None] * self.mat_y_base_dis +
            self.vec_scale_bspline[:, None] * self.mat_y_bspline_dis)

        # symbol
        self.flag_enable_symbol = False
        self.vec_name_func = numpy.array(['(x)'] * self.num_size, dtype=object)
        self.mat_para = torch.nn.Parameter(torch.zeros(self.num_size, 4, device=self.device)).requires_grad_(False)

        # prune
        self.vec_mask = torch.ones(self.num_size, device=self.device)


    def update_range_grid(self, method:str='lstsq'):
        """
        update range of b-spline grid points in each channels.

        """
        
        # update grid
        self.mat_range_grid = torch.cat(
            (self.mat_x.min(dim=1)[0].reshape(self.num_size, 1), 
             self.mat_x.max(dim=1)[0].reshape(self.num_size, 1)), dim=1)
        self.mat_grid = get_mat_grid(
            device=self.device,
            mat_range_grid=self.mat_range_grid, num_grid=self.num_grid, num_size=self.num_size)
        
        # update calculate matrix
        self.mat_basefunc = get_mat_basefunc(
            mat_x=self.mat_x, mat_grid=self.mat_grid, num_deg_k=self.num_deg_k, num_grid_ext=self.num_grid_ext)
        self.mat_coef = torch.nn.Parameter(
            get_mat_coef(
                device=self.device, 
                mat_basefunc=self.mat_basefunc, mat_y_bspline=self.mat_y_bspline, method=method))

        # update display matrix
        self.mat_x_dis = get_mat_grid(
            device=self.device, 
            mat_range_grid=self.mat_range_grid, num_grid=self.num_sample_dis, num_size=self.num_size)
        self.mat_basefunc_dis = get_mat_basefunc(
            mat_x=self.mat_x_dis, mat_grid=self.mat_grid, num_deg_k=self.num_deg_k, num_grid_ext=self.num_grid_ext)


    def update_num_grid(self, num_grid:int, method:str='lstsq'):
        """
        update number of b-spline grid points i.e. knots.

        """

        # update grid
        self.num_grid = num_grid
        self.mat_grid = get_mat_grid(
            mat_range_grid=self.mat_range_grid, num_grid=self.num_grid, num_size=self.num_size)
        
        # update calculate matrix
        self.mat_basefunc = get_mat_basefunc(
            mat_x=self.mat_x, mat_grid=self.mat_grid, num_deg_k=self.num_deg_k, num_grid_ext=self.num_grid_ext)
        self.mat_coef = torch.nn.Parameter(
            get_mat_coef(
                device=self.device, 
                mat_basefunc=self.mat_basefunc, mat_y_bspline=self.mat_y_bspline, method=method))
        
        # update display matrix
        self.mat_x_dis = get_mat_grid(
            mat_range_grid=self.mat_range_grid, num_grid=self.num_sample_dis, num_size=self.num_size)
        self.mat_basefunc_dis = get_mat_basefunc(
            mat_x=self.mat_x_dis, mat_grid=self.mat_grid, num_deg_k=self.num_deg_k, num_grid_ext=self.num_grid_ext)
        

    def update_mask(self, tol_mask=0.05, prop_enable=None):
        """
        update mask of b-spline base functions.

        """
        if prop_enable is not None:
            
            num_enable = int(prop_enable*self.vec_act.shape[0])
            num_enable = max(num_enable, 1)
            ind_sort = torch.argsort(self.vec_act, descending=True)
            ind_enbale = ind_sort[:num_enable]
            ind_disable = torch.ones_like(self.vec_act).to(bool)
            ind_disable[ind_enbale] = 0

        else:
            
            ind_disable = self.vec_act < tol_mask

        self.vec_mask[ind_disable] = 0


    def auto_symbol(
            self, 
            flag_enable_print_dyna:bool=True, 
            flag_enable_print_stat:bool=False, 
            str_print_prefix:str=''):

        for ii in range(self.num_size):

            if self.vec_mask[ii] == 1:

                vec_x = self.mat_x_dis[ii, :].detach()
                vec_y = self.mat_y_dis[ii, :].detach()

                list_name_func = []
                list_vec_para = []
                list_r2 = []
                for name_func in lib_symbol.keys():
                    
                    vec_para, r2, model = get_vec_para(
                        name_func=name_func, vec_x=vec_x, vec_y=vec_y, )
                    
                    list_name_func.append(name_func)
                    list_vec_para.append(vec_para)
                    list_r2.append(r2)

                    if flag_enable_print_dyna:
                        print('%s%-12s: %7.4f' % (str_print_prefix, name_func, r2)+' '*15, end='\r')
                    

                ind_best = list_r2.index(max(list_r2))
                name_func = list_name_func[ind_best]
                vec_para = list_vec_para[ind_best]
                r2 = list_r2[ind_best]

                self.vec_name_func[ii] = name_func
                with torch.no_grad():
                    self.mat_para[ii, :] = vec_para.detach()

                if flag_enable_print_dyna:
                    print('%s%-12s: %7.4f' % (str_print_prefix, name_func, r2)+' '*20)
                elif flag_enable_print_stat:
                    print('%s%-12s: %7.4f' % (str_print_prefix, name_func, r2))


    def manu_symbol(
            self, 
            ind_channel:int, 
            name_func:str, 
            list_vec_para_0=None):

        vec_x = self.mat_x_dis[ind_channel, :].detach()
        vec_y = self.mat_y_dis[ind_channel, :].detach()

        vec_para, r2, model = get_vec_para(
            name_func=name_func, vec_x=vec_x, vec_y=vec_y, 
            list_vec_para_0=list_vec_para_0)

        self.vec_name_func[ind_channel] = name_func
        with torch.no_grad():
            self.mat_para[ind_channel, :] = vec_para.detach()

        print('%-12s: %7.4f' % (name_func, r2))


    def enable_symbol(self):

        self.flag_enable_symbol = True
        self.mat_para.requires_grad_(True)

        self.flag_enable_bspline = False
        self.vec_scale_base.requires_grad_(False)
        self.vec_scale_bspline.requires_grad_(False)
        self.mat_coef.requires_grad_(False)


    def init_ax(self, scale_fig=1, list_name_input=None):

        # build figure
        len_rect = 1
        len_gap = 0.1
        width_net = self.num_size
        height_net = 1
        width_fig = (width_net+1) * len_rect + (width_net + 1) * len_gap
        height_fig = height_net * len_rect + (height_net + 1) * len_gap
        fig = matplotlib.pyplot.figure(
            figsize=scale_fig*numpy.array([width_fig, height_fig]))
        fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
        
        # build axes
        array_ax_edge_layer_flatten = numpy.empty(shape=(self.num_size), dtype=object)  
        for ii in range(self.num_size):
            ax_edge = fig.add_subplot(1, self.num_size, ii+1)
            ax_edge.set_xticks([])
            ax_edge.set_yticks([])
            array_ax_edge_layer_flatten[ii] = ax_edge
        
        return array_ax_edge_layer_flatten
    

    def prune(self, ind_node_keep_input, ind_node_keep_output):

        """
        :ind_node_keep_input: (edge.num_input, ) torch.Tensor[bool]
            index of input node.
        :ind_node_keep_output: (edge.num_output, ) torch.Tensor[bool]
            index of output node.     
        :ind_edge_keep: (num_size, ) torch.Tensor[bool]
            index of keep edge.
        """
        # (num_output, num_input) -> (num_size, )
        ind_edge_keep = torch.einsum(
            'i,j->ij', ind_node_keep_output, ind_node_keep_input).flatten()

        # update mats
        self.num_input = ind_node_keep_input.sum().item()
        self.num_output = ind_node_keep_output.sum().item()
        self.num_size = self.num_output * self.num_input 
        self.mat_range_grid = self.mat_range_grid[ind_edge_keep, :]
        self.mat_grid = self.mat_grid[ind_edge_keep, :]
        self.mat_basefunc = self.mat_basefunc[ind_edge_keep, :, :]
        self.mat_x = self.mat_x[ind_edge_keep, :]
        self.mat_y = self.mat_y[ind_edge_keep, :]
        self.mat_y_base = self.mat_y_base[ind_edge_keep, :]
        self.mat_y_bspline = self.mat_y_bspline[ind_edge_keep, :]
        
        self.vec_name_func = self.vec_name_func[ind_edge_keep.cpu().numpy()] # numpy.array
        self.vec_act = self.vec_act[ind_edge_keep]
        self.vec_mask = self.vec_mask[ind_edge_keep]

        self.mat_basefunc_dis = self.mat_basefunc_dis[ind_edge_keep, :, :]
        self.mat_x_dis = self.mat_x_dis[ind_edge_keep, :]
        self.mat_y_dis = self.mat_y_dis[ind_edge_keep, :]
        self.mat_y_base_dis = self.mat_y_base_dis[ind_edge_keep, :]
        self.mat_y_bspline_dis = self.mat_y_bspline_dis[ind_edge_keep, :]

        with torch.no_grad():
            self.vec_scale_base = torch.nn.Parameter(self.vec_scale_base[ind_edge_keep])
            self.vec_scale_bspline = torch.nn.Parameter(self.vec_scale_bspline[ind_edge_keep])
            self.mat_coef = torch.nn.Parameter(self.mat_coef[ind_edge_keep, :])
            self.mat_para = torch.nn.Parameter(self.mat_para[ind_edge_keep, :])
        


    def plot(
            self, 
            array_ax_edge_layer=None, 
            flag_enable_base=True, 
            flag_enable_bspline=True, 
            flag_enable_basefunc=True, 
            flag_enable_symbol=False, ):

        if array_ax_edge_layer is not None:
            # (num_output, num_input) ->  (1, num_size) -> (num_size, )
            array_ax_edge_layer_flatten = \
                array_ax_edge_layer.reshape(-1, self.num_size).squeeze(axis=0)
        else:
            array_ax_edge_layer_flatten = self.init_ax()

        # detach data
        mat_x_dis = self.mat_x_dis.detach().cpu().numpy()
        mat_y_base_dis = self.mat_y_base_dis.detach().cpu().numpy()
        mat_y_bspline_dis = self.mat_y_bspline_dis.detach().cpu().numpy()
        mat_coef = self.mat_coef.detach().cpu().numpy()
        mat_basefunc_dis = self.mat_basefunc_dis.detach().cpu().numpy()
        vec_scale_base = self.vec_scale_base.detach().cpu().numpy()
        vec_scale_bspline = self.vec_scale_bspline.detach().cpu().numpy()

        # (num_output, num_input) ->  (1, num_size) -> (num_size, )
        for ii in range(self.num_size):
            
            ax_edge:matplotlib.axes._axes.Axes = array_ax_edge_layer_flatten[ii]
            # mask tag
            if self.vec_mask[ii] == 0:
                ax_edge.scatter(
                    x=0.1, y=0.9, s=40, marker='o', color='C3', 
                    transform=ax_edge.transAxes)
            
            vec_x_dis = mat_x_dis[ii, :]
            vec_y_base_dis = mat_y_base_dis[ii, :] * vec_scale_base[ii]
            vec_y_bspline_dis = vec_y_base_dis + mat_y_bspline_dis[ii, :] * vec_scale_bspline[ii]

            # base
            if flag_enable_base:
                ax_edge.plot(
                    vec_x_dis, vec_y_base_dis, color='C0', linewidth=2, alpha=0.75)
            # bspline
            if flag_enable_bspline:      

                ax_edge.plot(
                    # C1 is normal; C0 is debug
                    vec_x_dis, vec_y_bspline_dis, color='C1', linewidth=2)
                
                if flag_enable_basefunc:

                    for jj in range(self.mat_basefunc_dis.shape[1]):

                        vec_y_basefunc_dis = (
                            vec_y_base_dis + 
                            mat_basefunc_dis[ii, jj, :]*mat_coef[ii, jj] * vec_scale_bspline[ii])
                        ax_edge.plot(
                            vec_x_dis, vec_y_basefunc_dis, linewidth=1, alpha=0.75)
                        
            if flag_enable_symbol:
                
                name_func = self.vec_name_func[ii]
                vec_x_dis = mat_x_dis[ii, :]
                vec_y_dis = get_vec_y_symbol(
                    name_func=name_func, 
                    vec_para=self.mat_para[ii, :].detach().cpu(), 
                    vec_x=vec_x_dis)
                ax_edge.plot(
                    vec_x_dis, vec_y_dis, color='C4', linewidth=2)
                ax_edge.text(
                    x=0, y=0.02, s=name_func, 
                    transform=ax_edge.transAxes, 
                    va='bottom', ha='left', 
                    color='C4')


    def forward(self, mat_input: torch.Tensor, flag_valid:bool=False):
        """
        forward function of KAN layer.

        """

        # (num_batch, num_input) -> (num_batch, num_output, num_input)
        mat_input = torch.einsum('ij,k->ikj', mat_input, torch.ones(self.num_output, device=self.device))

        # (num_batch, num_output, num_input) -> (num_batch, num_size) -> (num_size, num_batch)
        mat_x = mat_input.reshape(-1, self.num_size).T

        if self.flag_enable_bspline:

            # base part
            # (num_size, num_batch)
            mat_y_base = torch.nn.functional.selu(mat_x)

            # bspline part
            # (num_size, num_basefunc, num_batch)
            mat_basefunc = get_mat_basefunc(
                mat_x=mat_x, mat_grid=self.mat_grid, num_deg_k=self.num_deg_k, num_grid_ext=self.num_grid_ext)
            # (num_size, num_batch)
            mat_y_bspline = get_mat_y_bspline(
                mat_basefunc=mat_basefunc, mat_coef=self.mat_coef)

            # (num_size, num_batch)
            mat_y = (
                # [(num_size, ) -> (num_size, 1)] * (num_size, num_batch)
                self.vec_scale_base[:, None] * mat_y_base +
                self.vec_scale_bspline[:, None] * mat_y_bspline)
            
            # (num_size, num_batch) = (num_size, num_batch) * (num_size, 1)
            mat_y = mat_y * self.vec_mask[:, None]
            
            # update train data space
            self.mat_x = mat_x.detach()
            self.mat_y_base = mat_y_base.detach()
            self.mat_y_bspline = mat_y_bspline.detach()
            self.mat_y = mat_y.detach()
            self.mat_basefunc = mat_basefunc.detach()

            if not flag_valid:

                # update display data space 
                self.mat_y_base_dis = torch.nn.functional.selu(self.mat_x_dis)
                self.mat_y_bspline_dis = get_mat_y_bspline(
                    mat_basefunc=self.mat_basefunc_dis, mat_coef=self.mat_coef)
                self.mat_y_dis = (
                    self.vec_scale_base[:, None] * self.mat_y_base_dis +
                    self.vec_scale_bspline[:, None] * self.mat_y_bspline_dis)
            
        elif self.flag_enable_symbol:

            # (num_size, num_batch)
            mat_y = torch.zeros_like(mat_x)
            for ii in range(self.num_size):

                # (num_batch, )
                vec_x = mat_x[ii, :]
                name_func = self.vec_name_func[ii]
                # (4, )
                vec_para = self.mat_para[ii, :]
                # (num_batch, )
                vec_y = get_vec_y_symbol(
                    name_func=name_func, 
                    vec_para=vec_para, vec_x=vec_x)
                mat_y[ii, :] = vec_y

            self.mat_x_dis = get_mat_grid(
                device=self.device, 
                mat_range_grid=torch.cat(
                    (mat_x.detach().min(dim=1).values[:, None], 
                     mat_x.detach().max(dim=1).values[:, None]), dim=1), 
                num_grid=self.num_sample_dis, 
                num_size=self.num_size)
        
        # update train data space activate intensity
        # (num_size, num_batch) -> (num_size, )
        self.vec_act = self.mat_y.abs().mean(dim=1)
        self.vec_act = self.vec_act / self.vec_act.max()

        # (num_size, num_batch) -> (num_batch, num_size) -> (num_batch, num_output, num_input)
        mat_output = mat_y.T.reshape(-1, self.num_output, self.num_input)

        # (num_batch, num_output, num_input) -> (num_batch, num_output)
        mat_output = torch.sum(mat_output, dim=2)

        return mat_output


class KAN(torch.nn.Module):

    def __init__(
            self,
            list_num_node:list[int],
            num_grid:int=6, 
            device:str='cpu'):
        
        super().__init__()

        self.device = device

        self.list_num_node = list_num_node
        self.num_layer_node = len(list_num_node)
        self.num_layer_edge = self.num_layer_node - 1

        self.num_input = list_num_node[0]
        self.num_output = list_num_node[-1]
        
        self.num_grid = num_grid
        
        self.make_edge()
        

    def make_edge(self):

        # make edge layer
        list_edge = []
        list_num_edge = []
        for ii in range(self.num_layer_edge):
            
            edge = KAN_EDGE(
                num_input=self.list_num_node[ii],
                num_output=self.list_num_node[ii+1], 
                num_grid=self.num_grid, 
                device=self.device)

            list_edge.append(edge)
            list_num_edge.append(edge.num_size)

        self.list_num_edge = list_num_edge
        self.modulelist_edge = torch.nn.ModuleList(list_edge)
        self.vec_bias = torch.nn.Parameter(2 * torch.rand(self.num_layer_edge, device=self.device) - 1)



    def update_range_grid(self, method:str='lstsq'):

        for ii in range(self.num_layer_edge):

            edge:KAN_EDGE = self.modulelist_edge[ii]
            edge.update_range_grid(method=method)
            self.forward(mat_input=self.dict_dataset['train_x'])

        
    def update_num_grid(self, num_grid:int, method:str='lstsq'):

        for ii in range(self.num_layer_edge):

            edge:KAN_EDGE = self.modulelist_edge[ii]
            edge.update_num_grid(num_grid=num_grid, method=method)


    def update_mask(
            self, 
            tol_mask=None, 
            prop_enable=None, 
            sub_edge_disable=None, 
            sub_node_disable=None):
        
        """
        :sub_edge_disable: (3, ) list
            subscript of disable edge.
            [seq_layer_edge, seq_node_output, seq_node_input]
        :sub_node_disable: (2, ) list
            subscript of disable node.
            [seq_layer_node, seq_node]
        :ind_edge_disable: (num_output, num_input) <-> (num_size, ) torch.Tensor[bool]
            index of disable edge channel in mask.

        """

        if tol_mask is not None:
            # update each layer to check tol_mask
            for seq_layer_edge in range(self.num_layer_edge):
                edge:KAN_EDGE = self.modulelist_edge[seq_layer_edge]
                edge.update_mask(tol_mask=tol_mask)
        
        if prop_enable is not None:
            # update each layer to check prop_enable
            for seq_layer_edge in range(self.num_layer_edge):
                edge:KAN_EDGE = self.modulelist_edge[seq_layer_edge]
                edge.update_mask(prop_enable=prop_enable)


        if sub_edge_disable is not None:
            
            seq_layer_edge = sub_edge_disable[0]
            seq_node_output = sub_edge_disable[1]
            seq_node_input = sub_edge_disable[2]

            edge:KAN_EDGE = self.modulelist_edge[seq_layer_edge]
            ind_edge_disable = torch.zeros(
                size=(edge.num_output, edge.num_input), device=self.device).to(bool)
            ind_edge_disable[seq_node_output, seq_node_input] = True
            ind_edge_disable = ind_edge_disable.flatten()
            edge.vec_mask[ind_edge_disable] = 0

        if sub_node_disable is not None:

            seq_layer_node = sub_node_disable[0]
            seq_node = sub_node_disable[1]

            seq_layer_edge_input = seq_layer_node - 1 
            seq_node_output = seq_node
            if seq_layer_edge_input >= 0:
                edge:KAN_EDGE = self.modulelist_edge[seq_layer_edge_input]
                # (num_output, num_input)
                ind_edge_disable = torch.zeros(
                    size=(edge.num_output, edge.num_input), device=self.device).to(bool)
                ind_edge_disable[seq_node_output, :] = True
                # (num_output, num_input) -> # (num_size, )
                ind_edge_disable = ind_edge_disable.flatten()
                edge.vec_mask[ind_edge_disable] = 0
            
            seq_layer_edge_output = seq_layer_node
            seq_node_input = seq_node
            if seq_layer_edge_output <= self.num_layer_node:
                edge:KAN_EDGE = self.modulelist_edge[seq_layer_edge_output]
                # (num_output, num_input)
                ind_edge_disable = torch.zeros(
                    size=(edge.num_output, edge.num_input), device=self.device).to(bool)
                ind_edge_disable[:, seq_node_input] = True
                # (num_output, num_input) -> # (num_size, )
                ind_edge_disable = ind_edge_disable.flatten()
                edge.vec_mask[ind_edge_disable] = 0

        # when node all output channel is disabled, the node input channel will be disabled
        # from back to front
        for seq_layer_edge in range(self.num_layer_edge-2, -1, -1):
            
            edge_input:KAN_EDGE  = self.modulelist_edge[seq_layer_edge]
            edge_output:KAN_EDGE = self.modulelist_edge[seq_layer_edge+1]
            
            # output edge layer
            # (num_output, num_input)
            mat_mask_output = edge_output.vec_mask.reshape(
                edge_output.num_output, edge_output.num_input)
            # (num_output, num_input) -> (num_input, )
            ind_edge_disable = mat_mask_output.sum(dim=0) == 0

            # input edge layer
            # ind_edge_disable has the rule that:
            # (edge_output.num_input, ) = (edge_input.num_output, )
            # (num_output, num_input)
            mat_mask_input = edge_input.vec_mask.reshape(
                edge_input.num_output, edge_input.num_input)
            mat_mask_input[ind_edge_disable, :] = 0
            # (num_output, num_input) -> (num_input, )
            edge_input.vec_mask = mat_mask_input.flatten()


    def prune(self):

        # prune each layer that all output channel is disable
        # from front to back (never deal with the laster edge layer)
        list_ind_node_keep = []
        for ii in range(self.num_layer_edge-1):

            edge:KAN_EDGE = self.modulelist_edge[ii]
            # (num_size, ) -> (num_output, num_input)
            mat_mask = edge.vec_mask.reshape(edge.num_output, edge.num_input)
            # (num_output, num_input) -> (num_output, )
            ind_node_keep = mat_mask.sum(dim=1) != 0

            list_ind_node_keep.append(ind_node_keep)
            self.list_num_node[ii+1] = ind_node_keep.sum().item()

        print(list_ind_node_keep)
        # add first and last node info into list_ind_node_keep
        list_ind_node_keep = [
            torch.ones(self.num_input, device=self.device).to(bool), 
            *list_ind_node_keep, 
            torch.ones(self.num_output, device=self.device).to(bool), ]
        print(list_ind_node_keep)

        # rebuild each edge layer
        for ii in range(self.num_layer_edge):
            edge:KAN_EDGE = self.modulelist_edge[ii]
            ind_node_keep_input = list_ind_node_keep[ii]
            ind_node_keep_output = list_ind_node_keep[ii+1]
            edge.prune(
                ind_node_keep_input=ind_node_keep_input, 
                ind_node_keep_output=ind_node_keep_output)
            self.list_num_edge[ii] = \
                ind_node_keep_input.sum().item() * ind_node_keep_output.sum().item()
            

    def auto_symbol(
            self, 
            flag_enable_print_dyna:bool=True, 
            flag_enable_print_stat:bool=True, 
            str_print_prefix:str=''):

        for ii in range(self.num_layer_edge):
            edge:KAN_EDGE = self.modulelist_edge[ii]
            edge.auto_symbol(
                flag_enable_print_dyna=flag_enable_print_dyna, 
                flag_enable_print_stat=flag_enable_print_stat, 
                str_print_prefix=str_print_prefix, )


    def manu_symbol(
            self, 
            sub_edge:list, 
            name_func:str, 
            list_vec_para_0=None):
        
        seq_layer_edge = sub_edge[0]
        seq_node_output = sub_edge[1]
        seq_node_input = sub_edge[2]

        edge:KAN_EDGE = self.modulelist_edge[seq_layer_edge]

        ind_edge = torch.zeros(
            size=(edge.num_output, edge.num_input), device=self.device).to(bool)
        ind_edge[seq_node_output, seq_node_input] = True
        ind_edge = ind_edge.flatten()
        ind_edge = ind_edge.nonzero().item()

        edge.manu_symbol(
            ind_channel=ind_edge, name_func=name_func, 
            list_vec_para_0=list_vec_para_0)


    def enable_symbol(self):

        for ii in range(self.num_layer_edge):
            edge:KAN_EDGE = self.modulelist_edge[ii]
            edge.enable_symbol()


    def formula(self):

        list_str_formula_output_new = []
        for ii in range(self.num_layer_edge):
            edge:KAN_EDGE = self.modulelist_edge[ii]
            list_str_formula_output_old = list_str_formula_output_new
            list_str_formula_output_new = []
            for jj in range(edge.num_output):
                list_str_formula_input = []
                for kk in range(edge.num_input):

                    if ii != 0 :
                        str_var = list_str_formula_output_old[kk]
                    else:
                        str_var = '(x_%d)'%kk

                    seq_node_output = jj
                    seq_node_input = kk
                    ind_edge = torch.zeros(
                        size=(edge.num_output, edge.num_input), device=self.device).to(bool)
                    ind_edge[seq_node_output, seq_node_input] = True
                    ind_edge = ind_edge.flatten()
                    ind_edge = ind_edge.nonzero().item()

                    vec_para = edge.mat_para[ind_edge, :].detach()
                    name_func = edge.vec_name_func[ind_edge]
                    str_part_inner = '%.4f*(x)+%.4f' % (vec_para[0].item(), vec_para[1].item())
                    str_part_outer = '(%.4f*%s+%.4f)' % (vec_para[2].item(), name_func,vec_para[3].item())
                    str_formula_input = str_part_outer.replace('(x)', '(%s)' %str_part_inner)
                    str_formula_input = str_formula_input.replace('(x)', '%s' % str_var)

                    list_str_formula_input.append(str_formula_input)

                str_formula_output = '+'.join(list_str_formula_input)
                str_formula_output = '(%s+%.4f)' % (str_formula_output, self.vec_bias[ii].item())
                list_str_formula_output_new.append(str_formula_output)
        
        str_formula = str_formula_output.replace('+-', '-')

        return str_formula



    def init_ax(
            self, 
            scale_fig=1, 
            list_name_input=None, 
            flag_enable_large=False, 
            flag_enable_index_edge=True, 
            flag_enable_index_node=True):

        # initial figure posi info
        len_rect = 1
        len_gap = 0.1
        
        if flag_enable_large:

            # build figure
            width_fig = 0
            height_fig = 0
            for ii in range(self.num_layer_edge):
                edge:KAN_EDGE = self.modulelist_edge[ii]
                width_fig = max(width_fig, edge.num_input)
                height_fig = height_fig + edge.num_output + 0.5
            fig = matplotlib.pyplot.figure(
                figsize=scale_fig*numpy.array([width_fig, height_fig]))

            # build main axes
            # ax_main = fig.add_axes(rect=(0.05, 0.05, 0.9, 0.9))
            ax_main = fig.add_axes(rect=(0.04, 0.04, 1.0, 1.0))
            ax_main.axis('equal')
            ax_main.axis('off')
            ax_main.set_xlim(numpy.array([-0.5, width_fig])*(len_rect+len_gap))
            ax_main.set_ylim(numpy.array([-0.5, height_fig])*(len_rect+len_gap))

            # build alpha
            array_alpha_edge = numpy.empty(shape=(self.num_layer_edge), dtype=object)
            for ii in range(self.num_layer_edge):
                edge:KAN_EDGE = self.modulelist_edge[ii]
                # (num_size, )
                array_alpha_edge_ii = edge.vec_act.cpu().numpy() * edge.vec_mask.cpu().numpy()
                # (num_size, ) -> (num_output, num_input)
                array_alpha_edge_ii = array_alpha_edge_ii.reshape(edge.num_output, edge.num_input)
                # save
                array_alpha_edge[ii] = array_alpha_edge_ii
            
            

            # plot edge
            # (num_layer, num_output, num_input) object array
            array_ax_edge_main = numpy.empty(shape=(self.num_layer_edge), dtype=object)
            array_posi_edge = numpy.empty(shape=(self.num_layer_edge), dtype=object)  
            # [ii, jj, kk] -> [seq_layer_edge, seq_node_output, seq_node_input]
            posi_edge_y_bias = 0
            for ii in range(self.num_layer_edge):
                edge:KAN_EDGE = self.modulelist_edge[ii]
                ax_main.text(
                    x=-0.6*len_rect,
                    y=posi_edge_y_bias*(len_rect + len_gap)-0.6*len_rect,
                    s='%d' % ii, c='C0',
                    va='top', ha='right')

                num_edge = self.list_num_edge[ii]
                num_node_input = self.list_num_node[ii]
                num_node_output = self.list_num_node[ii+1]
                # (num_output, num_input)
                arrray_ax_edge_jjkk = numpy.empty(shape=(edge.num_output, edge.num_input), dtype=object)
                array_posi_edge_jjkk = numpy.empty(shape=(edge.num_output, edge.num_input), dtype=object)
                for jj in range(num_node_output):    # num_output

                    posi_edge_y = (jj + posi_edge_y_bias) * (len_rect + len_gap)

                    ax_main.text(
                        x=-0.1*len_rect-0.5*len_rect,
                        y=posi_edge_y,
                        s='%d' % jj,
                        va='center', ha='right')

                    for kk in range(num_node_input): # num_input

                        posi_edge_x = kk * (len_rect+len_gap)
                        posi_edge = numpy.array([posi_edge_x, posi_edge_y])
                        
                        posi_edge_ax_lb = fig.transFigure.inverted().transform(
                            ax_main.transData.transform(posi_edge))
                        posi_edge_ax_ru = fig.transFigure.inverted().transform(
                            ax_main.transData.transform([posi_edge[0] + len_rect, posi_edge[1] + len_rect]))
                        ax_edge_x = float(posi_edge_ax_lb[0] - 0.5 * (posi_edge_ax_ru[0] - posi_edge_ax_lb[0]))
                        ax_edge_y = float(posi_edge_ax_lb[1] - 0.5 * (posi_edge_ax_ru[1] - posi_edge_ax_lb[1]))
                        ax_edge_w = float(posi_edge_ax_ru[0] - posi_edge_ax_lb[0])
                        ax_edge_h = float(posi_edge_ax_ru[1] - posi_edge_ax_lb[1])
                        ax_edge_rect = (ax_edge_x, ax_edge_y, ax_edge_w, ax_edge_h)

                        ax_edge = fig.add_axes(rect=ax_edge_rect)
                        ax_edge.set_facecolor('w')
                        matplotlib.pyplot.setp(
                            ax_edge.spines.values(), linewidth=1.5, 
                            alpha=array_alpha_edge[ii][jj][kk])
                        ax_edge.set_xticks([])
                        ax_edge.set_yticks([])

                        arrray_ax_edge_jjkk[jj][kk] = ax_edge
                        array_posi_edge_jjkk[jj][kk] = posi_edge

                        if jj == 0:
                            ax_main.text(
                                x=posi_edge_x,
                                y=posi_edge_y-0.6*len_rect,
                                s='%d' % kk,
                                va='top', ha='center')
                            if ii == 0 and list_name_input is not None:
                                if kk < len(list_name_input):
                                    in_features_name = list_name_input[kk]
                                    ax_main.text(
                                        x=posi_edge_x,
                                        y=posi_edge_y-0.8*len_rect,
                                        s='%s' % in_features_name,
                                        va='top', ha='center')

                posi_edge_y_bias += num_node_output + 0.5        
                array_ax_edge_main[ii] = arrray_ax_edge_jjkk         
                array_posi_edge[ii] = array_posi_edge_jjkk

        else:

            # build figure
            width_net = max(self.list_num_edge)
            height_net = len(self.list_num_node) + len(self.list_num_edge)
            width_fig = (width_net+1) * len_rect + (width_net + 1) * len_gap
            height_fig = height_net * len_rect + (height_net + 1) * len_gap
            fig = matplotlib.pyplot.figure(
                figsize=scale_fig*numpy.array([width_fig, height_fig]))

            # build main axes
            ax_main = fig.add_axes(rect=(0.05, 0.05, 0.9, 0.9))
            ax_main.axis('equal')
            ax_main.axis('off')
            ax_main.set_xlim([0, width_fig])
            ax_main.set_ylim([0, height_fig])

            # build alpha
            array_alpha_edge = numpy.empty(shape=(self.num_layer_edge), dtype=object)
            for ii in range(self.num_layer_edge):
                edge:KAN_EDGE = self.modulelist_edge[ii]
                # (num_size, )
                array_alpha_edge_ii = edge.vec_act.cpu().numpy() * edge.vec_mask.cpu().numpy()
                # (num_size, ) -> (num_output, num_input)
                array_alpha_edge_ii = array_alpha_edge_ii.reshape(edge.num_output, edge.num_input)
                # save
                array_alpha_edge[ii] = array_alpha_edge_ii

            # plot node
            height_bias = len_gap + 0.5 * len_rect
            # (num_layer+1, num_node) object array
            array_posi_node = numpy.empty(shape=(self.num_layer_edge+1), dtype=object)  
            for ii in range(self.num_layer_node):
                num_node = self.list_num_node[ii]
                # (num_node, ) object array
                array_posi_node_jj = numpy.empty(shape=(num_node), dtype=object)  
                for jj in range(num_node):
                    posi_node = [
                        (jj + 1) / (num_node + 1) * width_fig, height_bias]
                    array_posi_node_jj[jj] = posi_node
                    ax_main.scatter(
                        posi_node[0], posi_node[1], 40, 'k', 'o')
                    if flag_enable_index_node:
                        ax_main.text(
                            x=posi_node[0], y=posi_node[1], s='(%d,%d)' % (ii, jj),
                            c='k', ha='left', va='bottom')
                    if ii == 0 and list_name_input is not None:
                        if jj < len(list_name_input):
                            name_input = list_name_input[jj]
                            ax_main.text(
                                x=posi_node[0], y=posi_node[1]-0.3*len_rect,
                                s='%s' % name_input,
                                va='bottom', ha='center')
                array_posi_node[ii] = array_posi_node_jj
                height_bias = height_bias + 2*(len_rect + len_gap)

            # plot edge
            height_bias = 2 * len_gap + 1.5 * len_rect
            # (num_layer, num_output, num_input) object array
            array_ax_edge_main = numpy.empty(shape=(self.num_layer_edge), dtype=object)
            array_posi_edge = numpy.empty(shape=(self.num_layer_edge), dtype=object)  
            # [ii, jj, kk] -> [seq_layer_edge, seq_node_output, seq_node_input]
            for ii in range(self.num_layer_edge):  
                edge = self.modulelist_edge[ii]
                num_edge = self.list_num_edge[ii]
                num_node_input = self.list_num_node[ii]
                num_node_output = self.list_num_node[ii+1]
                # (num_output, num_input)
                arrray_ax_edge_jjkk = numpy.empty(shape=(edge.num_output, edge.num_input), dtype=object)
                array_posi_edge_jjkk = numpy.empty(shape=(edge.num_output, edge.num_input), dtype=object)
                for jj in range(num_node_output):    # num_output
                    for kk in range(num_node_input): # num_input
                        posi_edge = numpy.array([(jj+kk*num_node_output + 1) / (num_edge + 1) * width_fig, height_bias])
                        posi_edge_ax_lb = fig.transFigure.inverted().transform(
                            ax_main.transData.transform(posi_edge))
                        posi_edge_ax_ru = fig.transFigure.inverted().transform(
                            ax_main.transData.transform([posi_edge[0] + len_rect, posi_edge[1] + len_rect]))
                        ax_edge_x = float(posi_edge_ax_lb[0] - 0.5 * (posi_edge_ax_ru[0] - posi_edge_ax_lb[0]))
                        ax_edge_y = float(posi_edge_ax_lb[1] - 0.5 * (posi_edge_ax_ru[1] - posi_edge_ax_lb[1]))
                        ax_edge_w = float(posi_edge_ax_ru[0] - posi_edge_ax_lb[0])
                        ax_edge_h = float(posi_edge_ax_ru[1] - posi_edge_ax_lb[1])
                        ax_edge_rect = (ax_edge_x, ax_edge_y, ax_edge_w, ax_edge_h)
                        if flag_enable_index_edge:
                            ax_main.text(
                                x=posi_edge[0]+0.5*len_rect, y=posi_edge[1]+0.5*len_rect,
                                s='(%d,%d,%d)' % (ii, jj, kk),
                                c='k', ha='right', va='bottom')
                        ax_edge = fig.add_axes(rect=ax_edge_rect)
                        matplotlib.pyplot.setp(
                            ax_edge.spines.values(), linewidth=1.5, 
                            alpha=array_alpha_edge[ii][jj][kk])
                        ax_edge.set_xticks([])
                        ax_edge.set_yticks([])
                        arrray_ax_edge_jjkk[jj][kk] = ax_edge
                        array_posi_edge_jjkk[jj][kk] = posi_edge
                array_ax_edge_main[ii] = arrray_ax_edge_jjkk
                array_posi_edge[ii] = array_posi_edge_jjkk
                height_bias = height_bias + 2 * (len_rect + len_gap)

            # plot connection between node and edge
            for ii in range(self.num_layer_edge):
                # node_in
                num_node_input = self.list_num_node[ii]
                num_node_output = self.list_num_node[ii+1]
                for jj in range(num_node_output):  # out_features
                    posi_node_out = array_posi_node[ii+1][jj]
                    for kk in range(num_node_input):  # in_features
                        posi_node_in = array_posi_node[ii][kk]
                        posi_edge = array_posi_edge[ii][jj][kk]
                        ax_main.plot(
                            [posi_node_in[0], posi_edge[0]], 
                            [posi_node_in[1], posi_edge[1] - 0.5 * len_rect], 
                            color='k', linestyle='-', linewidth=1.5, 
                            alpha=array_alpha_edge[ii][jj][kk])
                        ax_main.plot(
                            [posi_node_out[0], posi_edge[0]], 
                            [posi_node_out[1], posi_edge[1] + 0.5 * len_rect], 
                            color='k', linestyle='-', linewidth=1.5, 
                            alpha=array_alpha_edge[ii][jj][kk])

        return fig, ax_main, array_ax_edge_main


    def plot(
            self, 
            scale_fig=1.0, 
            fig=None, 
            flag_enable_base=True,
            flag_enable_bspline=True, 
            flag_enable_basefunc=True, 
            flag_enable_symbol=False, 
            flag_enable_large=False, 
            list_name_input=None, 
            flag_enable_index_edge=True, 
            flag_enable_index_node=True):

        if fig is None:

            fig, ax_main, array_ax_edge_main = self.init_ax(
                scale_fig=scale_fig, 
                list_name_input=list_name_input, 
                flag_enable_large=flag_enable_large, 
                flag_enable_index_edge=flag_enable_index_edge, 
                flag_enable_index_node=flag_enable_index_node)

        for ii in range(self.num_layer_edge):

            edge:KAN_EDGE = self.modulelist_edge[ii]
            edge.plot(
                array_ax_edge_layer=array_ax_edge_main[ii], 
                flag_enable_base=flag_enable_base,
                flag_enable_bspline=flag_enable_bspline, 
                flag_enable_basefunc=flag_enable_basefunc, 
                flag_enable_symbol=flag_enable_symbol, )
            
        return fig


    def forward(self, mat_input: torch.Tensor, flag_valid:bool=False):
                
        mat_iter = mat_input

        for ii, edge in enumerate(self.modulelist_edge):
            mat_iter = edge.forward(mat_input=mat_iter, flag_valid=flag_valid) + self.vec_bias[ii]
        
        mat_output = mat_iter

        return mat_output
    

    def parity(self):

        # parity
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot()
        ax.set_aspect(1.0)

        merge_x = self.dict_dataset['merge_x']
        merge_y = self.dict_dataset['merge_y']
        train_x = self.dict_dataset['train_x']
        train_y = self.dict_dataset['train_y']
        valid_x = self.dict_dataset['valid_x']
        valid_y = self.dict_dataset['valid_y']

        with torch.no_grad():
            train_y_pred = self.forward(train_x)
            if valid_x is not None:
                valid_y_pred = self.forward(valid_x)
            else:
                valid_y = []
                valid_y_pred = []

        ax.plot(
            [merge_y.min(), merge_y.max()], 
            [merge_y.min(), merge_y.max()], 'k-', alpha=0.2, zorder=0)
        ax.scatter(train_y, train_y_pred, 10, color='C0', label='train', zorder=1)
        ax.scatter(valid_y, valid_y_pred, 10, color='C1', label='valid', zorder=1)
        ax.legend(frameon=False)


    def histogram_act(self):

        # vec_act histogram
        vec_act_all = torch.tensor([])
        for ii in range(self.num_layer_edge):

            edge:KAN_EDGE = self.modulelist_edge[ii]
            vec_act_all = torch.cat([vec_act_all, edge.vec_act])


        fig = matplotlib.pyplot.figure(figsize=(6, 3))
        ax = fig.add_subplot()
        ax.hist(vec_act_all[vec_act_all != 0], bins=25)
        

    def train_model(
            self, 
            dict_dataset:dict, 
            learn_rate=0.03, 
            num_epoch=200, 
            flag_enable_print_dyna:bool=True, 
            flag_enable_print_stat:bool=True):

        self.dict_dataset = dict_dataset
        train_x = dict_dataset['train_x']
        train_y = dict_dataset['train_y']
        valid_x = dict_dataset['valid_x']
        valid_y = dict_dataset['valid_y']

        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)
        criterion = torch.nn.MSELoss(reduction='mean')

        list_loss_train = []
        list_loss_valid = []
        for ii in range(num_epoch):
            
            optimizer.zero_grad()
            loss_train = criterion(self.forward(train_x), train_y)
            loss_train.backward()
            optimizer.step()
            loss_train = loss_train.item()
            
            loss_valid = 0
            if valid_x is not None:
                with torch.no_grad():
                    loss_valid = criterion(self.forward(valid_x, flag_valid=True), valid_y)
                    loss_valid = loss_valid.item()
                
            if flag_enable_print_dyna:
                str_end = '\n' if ii==num_epoch-1 else '\r'
                print('(%4d/%4d) train: %10.4e, valid: %10.4e' 
                    % (ii+1, num_epoch,  loss_train, loss_valid), end=str_end)
            elif flag_enable_print_stat:
                if (ii+1) % int(num_epoch/5) == 0:
                    print('(%4d/%4d) train: %10.4e, valid: %10.4e' 
                        % (ii+1, num_epoch,  loss_train, loss_valid))
                
            list_loss_train.append(loss_train)
            list_loss_valid.append(loss_valid)

        return list_loss_train, list_loss_valid
    

    def estimate(self):

        train_x = self.dict_dataset['train_x']
        train_y = self.dict_dataset['train_y']
        valid_x = self.dict_dataset['valid_x']
        valid_y = self.dict_dataset['valid_y']
        merge_y_std = self.dict_dataset['merge_y_std']

        with torch.no_grad():

            train_y_pred = self.forward(train_x, flag_valid=True)
            train_rmse = torch.sqrt(((train_y_pred - train_y)**2).mean()) * merge_y_std # eV
            train_mae = torch.abs(train_y_pred - train_y).mean() * merge_y_std # eV
            trian_ssr = ((train_y_pred - train_y.mean())**2).mean()
            trian_sse = ((train_y_pred - train_y)**2).mean()
            trian_sst = trian_ssr + trian_sse
            train_r2 = 1 - trian_sse / trian_sst

            valid_y_pred = self.forward(valid_x, flag_valid=True)
            valid_rmse = torch.sqrt(((valid_y_pred - valid_y)**2).mean()) * merge_y_std # eV
            valid_mae = torch.abs(valid_y_pred - valid_y).mean() * merge_y_std # eV
            valid_ssr = ((valid_y_pred - valid_y.mean())**2).mean()
            valid_sse = ((valid_y_pred - valid_y)**2).mean()
            valid_sst = valid_ssr + valid_sse
            valid_r2 = 1 - valid_sse / valid_sst

        return train_rmse, train_mae, train_r2, valid_rmse, valid_mae, valid_r2

if __name__ == '__main__':

    # # init layer
    # num_input = 2
    # num_output = 1  # must be one
    # num_grid = 10

    # kan = KAN(
    #     num_input=num_input, 
    #     num_output=num_output,
    #     list_num_node=[1], 
    #     num_grid=num_grid, )


    ###############################################
    # test device
    device = 'cuda'
    num_batch = 100
    num_input = 2
    num_output = 1  # must be one
    mat_input = torch.rand(num_batch, num_input, device=device)

    edge = KAN_EDGE(
        num_input=num_input, 
        num_output=num_output,
        device=device)
    edge.forward(mat_input)

    kan = KAN(
        list_num_node = [num_input, 10, num_output], 
        device=device)
    kan.forward(mat_input)
    ###############################################

[user_00@A7950 pymodel_v2]$ \Supporting Information
