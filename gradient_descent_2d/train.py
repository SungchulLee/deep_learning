from global_name_space import ARGS

def train(x0=2, y0=4):
    x0_trace = [x0]
    y0_trace = [y0]
    z0_trace = [ARGS.compute_loss(x0,y0)]
    for i in range(ARGS.epochs):
        grad_x, grad_y = ARGS.compute_gradient(x0_trace[-1],y0_trace[-1])
        x0, y0 = ARGS.apply_gradient_descent((x0_trace[-1],y0_trace[-1]), (grad_x,grad_y), ARGS.lr)
        x0_trace.append(x0)
        y0_trace.append(y0)
        z0_trace.append(ARGS.compute_loss(x0,y0))
        if abs(x0_trace[-1] - x0_trace[-2]) + abs(y0_trace[-1] - y0_trace[-2]) < 1e-8:
            break
    else: # no break
        print('Failed to converge.')
        print('Try new initial point.')
        print('You may adjust your learning rate lr.')
        print('You may increase your n_steps.')
    return x0_trace, y0_trace