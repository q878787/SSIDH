import torch


# 参考：https://blog.csdn.net/winycg/article/details/104410525
class Htanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        # input[input >= 1] = input[input >= 1].sign()
        # input[input <= -1] = input[input <= -1].sign()

        #input = input.sign()
        return input.tanh()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        grd = (1 - torch.tanh(input) ** 2)
        #input = input.tanh()

        #方法1：2021cvpr的RDH用t控制
        # t = 0.99
        # condition = (input.abs()>t) & (input.sign() == (grad_output*grd).sign())
        # #print(condition.sum(),input.abs().mean())
        # grd[condition] = grd[condition]/ (1 - t ** 2)

        #方法2：用平方控制
        #方向相反时
        condition = (input.sign() == (grad_output*grd).sign())
        #w1 = len(input)*input.shape[1]/condition.sum()
        grd[condition] = (-(1-input[condition]**2)+2)
        #方向相同时
        # condition2 = (input.sign() != (grad_output*grd).sign())
        # w2 = len(input)*input.shape[1]/condition2.sum()
        # grd[condition2] = w2*grd[condition2]

        #print(condition.sum(),w1,condition2.sum(),w2)

        return grad_output * grd, None


if __name__ == '__main__':
    htanh = Htanh()
    x1 = torch.tensor([0.9, 0.8, -0.5], requires_grad=True)
    x2 = torch.tensor([1., 1., -1.], requires_grad=True)
    y = htanh.apply(x1, x2)
    print(y)
    y.sum().backward()
    print(x1.grad)

