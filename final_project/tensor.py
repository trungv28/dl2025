from copy import copy, deepcopy


class Tensor:
    def __init__(self, data):
        self.data = data

    @classmethod
    def fill(cls, *dims, val=0):
        out = val
        for dim in reversed(dims):
            out = [deepcopy(out) for _ in range(dim)]
        return cls(out)

    @property
    def shape(self):
        dims = []
        data = self.data
        while isinstance(data, list):
            dims.append(len(data))
            data = data[0]
        return dims

    def flatten(self):
        def recur(data):
            if isinstance(data, list):
                out = []
                for sub_data in data:
                    out.extend(recur(sub_data))
                return out
            return [data]

        return Tensor(recur(self.data))

    def reshape(self, *dims):
        flat = copy(self.flatten().data)

        def recur(dims, flat):
            if len(dims) == 1:
                out = []
                for _ in range(dims[0]):
                    out.append(flat.pop(0))
                return out
            out = []
            for _ in range(dims[0]):
                out.append(recur(dims[1:], flat))
            return out

        return Tensor(recur(list(dims), flat))

    def __apply_elemwise(self, func, other=None):
        flat = self.flatten().data
        if other is None:
            out = [func(val) for val in flat]
        elif isinstance(other, Tensor):
            other_flat = other.flatten().data
            out = [func(val, other_val) for val, other_val in zip(flat, other_flat)]
        else:
            out = [func(val, other) for val in flat]
        return Tensor(out).reshape(*self.shape)

    def __add__(self, other):
        return self.__apply_elemwise(lambda a, b: a + b, other)

    def __sub__(self, other):
        return self.__apply_elemwise(lambda a, b: a - b, other)

    def __mul__(self, other):
        return self.__apply_elemwise(lambda a, b: a * b, other)

    def transpose2d(self):
        return Tensor([list(row) for row in zip(*self.data)])

    # only to transpose first two dimensions
    def T(self):
        dim1, dim2, dim3, dim4 = self.shape
        out = Tensor.fill(dim2, dim1, dim3, dim4, val=0).data
        for d1 in range(dim1):
            for d2 in range(dim2):
                out[d2][d1] = deepcopy(self.data[d1][d2])
        return Tensor(out)

    def activate(self, func):
        return self.__apply_elemwise(func)

    #simplify
    def dot(self, other):
        data_a, data_b = self.data, other.data
        shape_a, shape_b = self.shape, other.shape

        if len(shape_a) == 1 and len(shape_b) == 1:
            return sum(data_a[i] * data_b[i] for i in range(shape_a[0]))

        if len(shape_a) == 2 and len(shape_b) == 2:
            return Tensor(
                [
                    [
                        sum(data_a[i][k] * data_b[k][j] for k in range(shape_a[1]))
                        for j in range(shape_b[1])
                    ]
                    for i in range(shape_a[0])
                ]
            )

        if len(shape_a) == 3 and len(shape_b) == 3:
            return Tensor(
                [
                    [
                        [
                            sum(
                                data_a[batch][i][k] * data_b[batch][k][j]
                                for k in range(shape_a[2])
                            )
                            for j in range(shape_b[2])
                        ]
                        for i in range(shape_a[1])
                    ]
                    for batch in range(shape_a[0])
                ]
            )


    def add_pad(self, pad_size=1, pad_val=0):
        def add_pad2d(data):
            n, m = len(data), len(data[0])
            out = [[pad_val] * (m + 2 * pad_size) for _ in range(n + 2 * pad_size)]
            for i in range(n):
                for j in range(m):
                    out[i + pad_size][j + pad_size] = data[i][j]
            return out

        def recur(data, depth):
            if isinstance(data[0][0], (int, float)):
                return add_pad2d(data)
            else:
                return [recur(sub_data, depth + 1) for sub_data in data]

        return Tensor(recur(self.data, 0))

    def unpad(self, pad_size=1):
        batch, ch, p_h, p_w = self.shape
        h = p_h - 2 * pad_size
        w = p_w - 2 * pad_size

        out = Tensor.fill(batch, ch, h, w, val=0).data

        for b in range(batch):
            for c in range(ch):
                for i in range(h):
                    for j in range(w):
                        out[b][c][i][j] = self.data[b][c][i + pad_size][j + pad_size]
        return Tensor(out)

    def flip(self):
        def recur(data, depth):
            if isinstance(data[0][0], (int, float)):
                return [list(reversed(row)) for row in reversed(data)]
            else:
                return [recur(sub_data, depth + 1) for sub_data in data]

        return Tensor(recur(self.data, 0))

    def dilate(self, dilation=2):
        kernel = self.data
        out_ch, in_ch, k_h, k_w = self.shape

        new_h = k_h + (k_h - 1) * (dilation - 1)
        new_w = k_w + (k_w - 1) * (dilation - 1)
        dilated_kernel = [
            [
                [[0 for _ in range(new_w)] for _ in range(new_h)]
                for _ in range(in_ch)
            ]
            for _ in range(out_ch)
        ]
        for o_ch in range(out_ch):
            for i_ch in range(in_ch):
                for i in range(0, new_h, dilation):
                    for j in range(0, new_w, dilation):
                        src_i = i // dilation
                        src_j = j // dilation
                        if src_i < k_h and src_j < k_w:
                            dilated_kernel[o_ch][i_ch][i][j] = kernel[o_ch][i_ch][
                                src_i
                            ][src_j]
        return Tensor(dilated_kernel)

    def __apply_window(self, kernel_size, stride, window_func, out_ch):
        data = self.data
        batch, _, h, w = self.shape
        k_h = k_w = kernel_size
        o_h = (h - k_h) // stride + 1
        o_w = (w - k_w) // stride + 1

        out = Tensor.fill(batch, out_ch, o_h, o_w, val=0).data

        for b in range(batch):
            for c in range(out_ch):
                oi = 0
                for i in range(0, h - k_h + 1, stride):
                    oj = 0
                    for j in range(0, w - k_w + 1, stride):
                        out[b][c][oi][oj] = window_func(
                            data, b, c, i, j, k_h, k_w, stride
                        )
                        oj += 1
                    oi += 1
        return Tensor(out)

    def conv(self, kernel, stride=1):
        out_ch, in_ch, k_h, k_w = kernel.shape

        def conv_window(data, b, c, i, j, k_h, k_w, stride):
            val = 0
            for i_ch in range(in_ch):
                window = [
                    [data[b][i_ch][i + ki][j + kj] for kj in range(k_w)]
                    for ki in range(k_h)
                ]
                flat_win = Tensor(window).flatten().data
                flat_kernel = Tensor(kernel.data[c][i_ch]).flatten().data
                val += sum(w * k for w, k in zip(flat_win, flat_kernel))
            return val

        return self.__apply_window(
            kernel_size=k_h, stride=stride, window_func=conv_window, out_ch=out_ch
        )

    def max_pool(self, kernel_size=2, stride=2):
        def max_pool_window(data, b, c, i, j, k_h, k_w, stride):
            window = [
                [data[b][c][i + ki][j + kj] for kj in range(k_w)]
                for ki in range(k_h)
            ]
            flat_window = Tensor(window).flatten().data
            max_val = max(flat_window)
            mask = [1 if v == max_val else 0 for v in flat_window]
            return (max_val, mask)

        out = self.__apply_window(
            kernel_size=kernel_size,
            stride=stride,
            window_func=max_pool_window,
            out_ch=len(self.data[0]),
        )
        pooled = Tensor(
            [
                [[[cell[0] for cell in row] for row in ch] for ch in batch]
                for batch in out.data
            ]
        )
        masks = Tensor(
            [
                [[[cell[1] for cell in row] for row in ch] for ch in batch]
                for batch in out.data
            ]
        )
        return pooled, masks

    def avg_pool(self, kernel_size=2, stride=2):
        def avg_pool_window(data, b, c, i, j, k_h, k_w, stride):
            window = [
                [data[b][c][i + ki][j + kj] for kj in range(k_w)]
                for ki in range(k_h)
            ]
            vals = Tensor(window).flatten().data
            return sum(vals) / len(vals)

        return self.__apply_window(
            kernel_size=kernel_size,
            stride=stride,
            window_func=avg_pool_window,
            out_ch=len(self.data[0]),
        )

